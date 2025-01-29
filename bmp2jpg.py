import os
from PIL import Image

'''
    convert bmp to jpg

    purpose: ti에서 전달받은 bmp 이미지들 확인할 때 사용
'''
if __name__ == '__main__':
    # 변환할 BMP 파일들이 있는 디렉토리 경로
    directory_path = './datasets/smoke_test_data_well'
    save_directory_path = './datasets/smoke_test_data_well_jpg'

    os.makedirs(save_directory_path, exist_ok=True)

    # 디렉토리 내의 모든 파일을 확인
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.bmp'):  # BMP 파일만 선택
            bmp_path = os.path.join(directory_path, filename)
            jpg_path = os.path.join(save_directory_path, filename.replace('.bmp', '.jpg'))
            
            # BMP 파일을 열고 JPG로 변환 후 저장
            with Image.open(bmp_path) as img:
                img = img.convert('RGB')  # JPG는 RGB 모드여야 함
                img.save(jpg_path, 'JPEG')

            print(f'Converted: {filename} to {filename.replace(".bmp", ".jpg")}')
