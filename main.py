from tetration_divergence import TetrationDivergence
import multiprocessing

if __name__ == "__main__":
    # 'spawn' 방식 설정
    multiprocessing.set_start_method("spawn", force=True)

    # 전체 계산 범위 및 해상도 설정
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    nx_total = 1_000_000  # x 방향 해상도
    ny_total = 1_000_000  # y 방향 해상도
    output_file = "divergence_map.h5"

    # 테트레이션 발산 클래스 초기화
    tetration = TetrationDivergence()

    # HDF5 파일에 결과 저장
    tetration.save_to_hdf5(x_min, x_max, y_min, y_max, nx_total, ny_total, output_file)
