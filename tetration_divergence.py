import torch
import numpy as np
import h5py
import multiprocessing as mp
from multiprocessing import Manager
import os


class TetrationDivergence:
    def __init__(self, max_iter=500, escape_radius=1e+10):
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        self.gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.cpu_cores = mp.cpu_count()

        print(f"Detected GPUs: {self.gpus}")
        print(f"Detected CPU Cores: {self.cpu_cores}")

    def compute_chunk(self, x_min, x_max, y_min, y_max, nx_total, ny_total, x_start, x_end, y_start, y_end, gpu):
        """
        특정 청크에 대해 테트레이션 발산 계산 (GPU 선택 가능)
        """
        device = torch.device(gpu)
        dx = (x_max - x_min) / nx_total
        dy = (y_max - y_min) / ny_total

        while True:
            try:
                # 청크 범위의 x, y 좌표 계산
                x_indices = torch.arange(x_start, x_end, dtype=torch.float64, device=device)
                y_indices = torch.arange(y_start, y_end, dtype=torch.float64, device=device)
                x_chunk = x_min + x_indices * dx
                y_chunk = y_min + y_indices * dy

                # 복소수 그리드 생성
                c_real = x_chunk[:, None]
                c_imag = y_chunk[None, :]
                c = c_real + 1j * c_imag

                # 초기값 설정
                z = c.clone()
                divergence_map_chunk = torch.zeros((len(x_chunk), len(y_chunk)), dtype=torch.bool, device=device)

                for _ in range(self.max_iter):
                    z = c ** z
                    diverged = z.abs() > self.escape_radius
                    divergence_map_chunk |= diverged
                    z = torch.where(diverged, torch.full_like(z, self.escape_radius), z)

                return divergence_map_chunk.cpu().numpy().astype(np.uint8) * 255

            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA Out of Memory on GPU {gpu}: Reducing chunk size...")
                # 청크 크기를 줄임
                step_size = max(256, (x_end - x_start) // 2)
                x_end = x_start + step_size
                y_end = y_start + step_size

    def save_to_hdf5(self, x_min, x_max, y_min, y_max, nx_total, ny_total, output_file, max_memory_usage=0.3):
        """
        GPU 메모리에 맞게 동적으로 청크 크기를 계산하고 데이터 저장
        """
        gpu_memory_limits = []
        for gpu_id, gpu in enumerate(self.gpus):
            device_props = torch.cuda.get_device_properties(gpu_id)
            total_memory = device_props.total_memory
            available_memory = total_memory * max_memory_usage
            gpu_memory_limits.append(available_memory)

        # 최소 GPU 메모리에 기반한 초기 청크 크기 설정
        min_gpu_memory = min(gpu_memory_limits)
        bytes_per_pixel = 16  # 복소수(double)의 메모리 사용량
        max_chunk_pixels = int(min_gpu_memory / bytes_per_pixel)

        chunk_size = int(np.sqrt(max_chunk_pixels))  # 정사각형 청크
        chunk_nx = max(1024, min(nx_total, chunk_size))
        chunk_ny = max(1024, min(ny_total, chunk_size))

        print(f"Dynamic chunk size set to: {chunk_nx} x {chunk_ny}")

        # HDF5 파일 생성
        with h5py.File(output_file, "w") as f:
            f.create_dataset("divergence_map", shape=(nx_total, ny_total), dtype="uint8", compression="gzip")

        # 작업 청크 분할
        tasks = []
        manager = Manager()
        lock = manager.Lock()

        for x_start in range(0, nx_total, chunk_nx):
            x_end = min(x_start + chunk_nx, nx_total)
            for y_start in range(0, ny_total, chunk_ny):
                y_end = min(y_start + chunk_ny, ny_total)
                gpu = self.gpus[len(tasks) % len(self.gpus)]  # 순환적으로 GPU 선택
                tasks.append((x_min, x_max, y_min, y_max, nx_total, ny_total, x_start, x_end, y_start, y_end, output_file, gpu, lock))

        # 병렬 작업 실행
        with mp.Pool(processes=len(self.gpus)) as pool:
            pool.map(self._process_chunk, tasks)

    def _process_chunk(self, args):
        """
        병렬 작업을 위한 래퍼 함수
        """
        (x_min, x_max, y_min, y_max, nx_total, ny_total,
         x_start, x_end, y_start, y_end, output_file, gpu, lock) = args

        # 청크 계산
        result = self.compute_chunk(
            x_min, x_max, y_min, y_max, nx_total, ny_total, x_start, x_end, y_start, y_end, gpu
        )

        # 결과 크기 확인
        result_shape = result.shape
        expected_shape = (x_end - x_start, y_end - y_start)

        if result_shape != expected_shape:
            print(f"Warning: Result shape {result_shape} does not match expected shape {expected_shape}.")
            print(f"Adjusting stored region to match result size.")
            x_end = x_start + result_shape[0]
            y_end = y_start + result_shape[1]

        # HDF5 파일에 안전하게 저장
        with lock, h5py.File(output_file, "a") as f:
            f["divergence_map"][x_start:x_end, y_start:y_end] = result
