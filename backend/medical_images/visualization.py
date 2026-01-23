"""
의료 이미지 3D 시각화 모듈
Django media 경로의 PNG 이미지를 사용한 3D 시각화
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    logger.warning(f"일부 라이브러리가 없습니다: {e}")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly가 설치되지 않았습니다. pip install plotly를 실행하세요.")
        go = None
        make_subplots = None

try:
    from scipy import ndimage
    from skimage import measure, morphology
except ImportError:
    logger.warning("scikit-image가 설치되지 않았습니다.")
    ndimage = None
    measure = None
    morphology = None


def load_patient_images(patient_id, media_root=None):
    """
    Django media 경로에서 환자의 모든 이미지와 마스크를 로드하여 3D 볼륨으로 재구성
    
    Args:
        patient_id: 환자 ID (문자열)
        media_root: MEDIA_ROOT 경로 (None이면 settings에서 가져옴)
    
    Returns:
        volume: 3D numpy 배열 (z축은 시간순 또는 업로드 순)
        mask_volume: 3D 마스크 배열
        spacing: (x, y, z) 픽셀 간격 (기본값 사용)
        image_info: 각 슬라이스의 메타데이터 리스트
    """
    if media_root is None:
        media_root = settings.MEDIA_ROOT
    
    base_path = Path(media_root) / 'medical_images' / str(patient_id)
    images_dir = base_path / 'images'
    masks_dir = base_path / 'masks'
    
    if not images_dir.exists():
        logger.error(f"이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
        return None, None, None, None
    
    # 모든 날짜 폴더에서 이미지 파일 찾기
    image_files = []
    for date_dir in sorted(images_dir.rglob('*.png')):
        if date_dir.is_file():
            image_files.append(date_dir)
    
    # 마스크 파일 찾기 (images와 매칭)
    mask_files = []
    image_mask_pairs = []
    
    for img_file in image_files:
        # images/YYYY/MM/DD/filename.png -> masks/YYYY/MM/DD/filename_mask.png
        relative_path = img_file.relative_to(images_dir)
        mask_file = masks_dir / relative_path.parent / f"{relative_path.stem}_mask.png"
        
        if mask_file.exists():
            image_mask_pairs.append((img_file, mask_file))
        else:
            # 마스크가 없으면 빈 마스크 사용
            image_mask_pairs.append((img_file, None))
            logger.warning(f"마스크 파일을 찾을 수 없습니다: {mask_file}")
    
    if not image_mask_pairs:
        logger.error(f"이미지 파일을 찾을 수 없습니다: {images_dir}")
        return None, None, None, None
    
    logger.info(f"총 {len(image_mask_pairs)}개의 이미지-마스크 쌍 발견")
    
    # 촬영 날짜 순서로 정렬 (파일 경로의 날짜 정보 사용)
    def get_date_from_path(path):
        """경로에서 날짜 정보 추출 (YYYY/MM/DD)"""
        parts = path.parts
        try:
            # medical_images/patient_id/images/YYYY/MM/DD/filename.png
            # 또는 medical_images/patient_id/images/YYYY/MM/DD/filename.png
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 4:  # YYYY
                    if i + 2 < len(parts) and parts[i+1].isdigit() and parts[i+2].isdigit():
                        return (int(parts[i]), int(parts[i+1]), int(parts[i+2]))
        except:
            pass
        # 날짜를 찾을 수 없으면 파일 수정 시간 사용
        return (0, 0, 0)
    
    image_mask_pairs.sort(key=lambda x: (get_date_from_path(x[0]), x[0].name))
    
    # 3D 볼륨 생성
    slices = []
    masks = []
    image_info = []
    
    for idx, (img_file, mask_file) in enumerate(image_mask_pairs):
        try:
            # 원본 이미지 로드
            img = Image.open(img_file).convert('L')
            img_array = np.array(img)
            
            # 마스크 로드 (있으면)
            if mask_file and mask_file.exists():
                mask = Image.open(mask_file).convert('L')
                mask_array = np.array(mask)
                mask_array = (mask_array > 127).astype(np.uint8)
            else:
                # 마스크가 없으면 빈 마스크 생성
                mask_array = np.zeros_like(img_array, dtype=np.uint8)
            
            # 이미지 크기 통일 (첫 번째 이미지 크기에 맞춤)
            if idx == 0:
                target_size = img_array.shape
            else:
                if img_array.shape != target_size:
                    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
                    img_array = np.array(img)
                    if mask_array.shape != target_size:
                        mask = Image.fromarray(mask_array).resize((target_size[1], target_size[0]), Image.NEAREST)
                        mask_array = np.array(mask)
            
            slices.append(img_array)
            masks.append(mask_array)
            
            # 메타데이터 저장
            date_info = get_date_from_path(img_file)
            image_info.append({
                'index': idx,
                'filename': img_file.name,
                'date': date_info,
                'path': str(img_file.relative_to(Path(media_root)))
            })
            
        except Exception as e:
            logger.error(f"이미지 로드 실패 ({img_file}): {str(e)}")
            continue
    
    if not slices:
        logger.error("로드된 이미지가 없습니다.")
        return None, None, None, None
    
    # 3D 볼륨 생성 (z축은 시간순)
    volume = np.stack(slices, axis=2)
    mask_volume = np.stack(masks, axis=2)
    
    # 픽셀 간격 (일반적인 유방 MRI 값 사용)
    # z축 간격은 이미지 간 시간 간격을 의미 (예: 1mm)
    spacing = (0.5, 0.5, 1.0)  # mm
    
    logger.info(f"볼륨 크기: {volume.shape}")
    logger.info(f"픽셀 간격: {spacing} mm")
    logger.info(f"총 {len(slices)}개 슬라이스 로드 완료")
    
    return volume, mask_volume, spacing, image_info


def visualize_3d_volume_matplotlib(volume, spacing, num_slices=9):
    """
    matplotlib을 사용한 3D 볼륨 시각화 (슬라이스 뷰)
    """
    if volume is None:
        print("볼륨 데이터가 없습니다.")
        return
    
    # 중간 슬라이스들 선택
    z_indices = np.linspace(0, volume.shape[2]-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('PNG 3D 볼륨 - 슬라이스 뷰', fontsize=16, fontweight='bold')
    
    for idx, z in enumerate(z_indices):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(volume[:, :, z], cmap='gray')
        axes[row, col].set_title(f'슬라이스 {z}/{volume.shape[2]-1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('png_slices_view.png', dpi=150, bbox_inches='tight')
    print("슬라이스 뷰 이미지 저장: png_slices_view.png")
    plt.show()


def visualize_3d_volume_plotly(volume, spacing, mask_volume=None, threshold=0.3):
    """
    Plotly를 사용한 인터랙티브 3D 시각화 (메쉬 기반, 종양 포함)
    """
    if volume is None:
        print("볼륨 데이터가 없습니다.")
        return None
    
    fig = go.Figure()
    
    # 볼륨 정규화
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # 유방 조직 메쉬 생성
    threshold_value = volume_norm.max() * threshold
    try:
        # 종양 영역 제외하고 유방 조직만
        tissue_volume = volume_norm.copy()
        if mask_volume is not None:
            tissue_volume[mask_volume > 0] = 0  # 종양 영역 제거
        
        verts, faces, normals, values = measure.marching_cubes(
            tissue_volume, threshold_value, spacing=spacing
        )
        
        # 유방 조직 메시 추가
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            colorscale='Viridis',
            intensity=values,
            opacity=0.6,
            showscale=True,
            name='Breast Tissue'
        ))
    except Exception as e:
        print(f"유방 조직 메시 생성 실패: {e}")
    
    # 종양 메쉬 생성
    if mask_volume is not None:
        try:
            # 종양 마스크에 원본 볼륨 값 적용
            tumor_volume = np.zeros_like(volume_norm)
            tumor_volume[mask_volume > 0] = volume_norm[mask_volume > 0]
            
            if np.sum(tumor_volume > 0) > 0:
                tumor_threshold = tumor_volume.max() * 0.5
                verts_tumor, faces_tumor, normals_tumor, values_tumor = measure.marching_cubes(
                    tumor_volume, tumor_threshold, spacing=spacing
                )
                
                # 종양 메시 추가 (빨간색)
                fig.add_trace(go.Mesh3d(
                    x=verts_tumor[:, 0],
                    y=verts_tumor[:, 1],
                    z=verts_tumor[:, 2],
                    i=faces_tumor[:, 0],
                    j=faces_tumor[:, 1],
                    k=faces_tumor[:, 2],
                    color='red',
                    opacity=0.9,
                    name='Tumor'
                ))
                print(f"종양 메시 생성 완료: {len(verts_tumor)} vertices")
        except Exception as e:
            print(f"종양 메시 생성 실패: {e}")
    
    fig.update_layout(
        title='PNG 3D 볼륨 시각화 (인터랙티브) - 종양 포함',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            bgcolor='white'
        ),
        width=1000,
        height=800
    )
    
    return fig


def visualize_3d_volume_interactive_slices(volume, spacing):
    """
    인터랙티브한 슬라이스 탐색기 (Plotly 사용)
    """
    if volume is None:
        print("볼륨 데이터가 없습니다.")
        return None
    
    # 볼륨 정규화
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # 초기 슬라이스 (중간)
    initial_slice = volume.shape[2] // 2
    
    # 슬라이더를 사용한 인터랙티브 시각화
    fig = go.Figure()
    
    # 초기 이미지
    fig.add_trace(go.Heatmap(
        z=volume_norm[:, :, initial_slice],
        colorscale='gray',
        showscale=False
    ))
    
    # 슬라이더 추가
    steps = []
    for i in range(volume.shape[2]):
        step = dict(
            method='update',
            args=[{'z': [volume_norm[:, :, i]]}],
            label=f'슬라이스 {i}'
        )
        steps.append(step)
    
    sliders = [dict(
        active=initial_slice,
        currentvalue={"prefix": "슬라이스: "},
        steps=steps
    )]
    
    fig.update_layout(
        title='PNG 3D 볼륨 - 인터랙티브 슬라이스 탐색기',
        sliders=sliders,
        width=800,
        height=800
    )
    
    return fig


def visualize_3d_volume_plotly_voxel(volume, mask_volume, spacing, 
                                     downsample_factor=2, 
                                     tissue_threshold_percentile=20):
    """
    Plotly를 사용한 복셀 기반 3D 시각화 (점 마커)
    """
    if volume is None:
        print("볼륨 데이터가 없습니다.")
        return None
    
    # 다운샘플링 (성능 향상)
    if downsample_factor > 1:
        volume = volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
        if mask_volume is not None:
            mask_volume = mask_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # 볼륨 정규화 (0-255 범위)
    volume_min = volume.min()
    volume_max = volume.max()
    volume_norm = ((volume - volume_min) / (volume_max - volume_min) * 255).astype(np.uint8)
    
    # 색상 맵 생성 (보라색 → 녹색 → 노란색)
    colorscale = [
        [0.0, 'rgb(75, 0, 130)'],    # 어두운 보라색
        [0.3, 'rgb(0, 100, 0)'],     # 녹색
        [0.6, 'rgb(255, 255, 0)'],   # 노란색
        [1.0, 'rgb(255, 255, 255)']  # 흰색
    ]
    
    # 3D 볼륨 시각화
    fig = go.Figure()
    
    # 유방 조직 시각화 (임계값 이상인 voxel만 표시)
    threshold = np.percentile(volume_norm, tissue_threshold_percentile)
    tissue_mask = volume_norm > threshold
    
    # 종양 마스크가 있으면 유방 조직에서 제외
    if mask_volume is not None:
        tissue_mask = tissue_mask & (mask_volume == 0)
    
    # 복셀 좌표 생성
    x, y, z = np.where(tissue_mask)
    values = volume_norm[tissue_mask]
    
    # 스케일링 (실제 물리적 크기)
    x_scaled = x * spacing[0] * downsample_factor
    y_scaled = y * spacing[1] * downsample_factor
    z_scaled = z * spacing[2] * downsample_factor
    
    print(f"유방 조직 복셀 수: {len(x)}")
    
    # 유방 조직 산점도
    fig.add_trace(go.Scatter3d(
        x=x_scaled,
        y=y_scaled,
        z=z_scaled,
        mode='markers',
        marker=dict(
            size=2,
            color=values,
            colorscale=colorscale,
            opacity=0.6,
            showscale=True,
            colorbar=dict(title="Intensity", x=1.1)
        ),
        name='Breast Tissue',
        hovertemplate='X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<br>Intensity: %{marker.color}<extra></extra>'
    ))
    
    # 종양 마스크가 있으면 추가
    if mask_volume is not None:
        tumor_mask = mask_volume > 0
        x_tumor, y_tumor, z_tumor = np.where(tumor_mask)
        
        if len(x_tumor) > 0:
            x_tumor_scaled = x_tumor * spacing[0] * downsample_factor
            y_tumor_scaled = y_tumor * spacing[1] * downsample_factor
            z_tumor_scaled = z_tumor * spacing[2] * downsample_factor
            
            print(f"종양 복셀 수: {len(x_tumor)}")
            
            # 종양은 빨간색으로 강조 (더 크고 불투명하게)
            fig.add_trace(go.Scatter3d(
                x=x_tumor_scaled,
                y=y_tumor_scaled,
                z=z_tumor_scaled,
                mode='markers',
                marker=dict(
                    size=5,  # 크기 증가 (3 -> 5)
                    color='red',
                    opacity=1.0,  # 완전 불투명
                    line=dict(width=0)  # 테두리 없음
                ),
                name='Tumor',
                hovertemplate='X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<br>종양<extra></extra>'
            ))
    
    # 레이아웃 설정
    fig.update_layout(
        title="DICOM 3D 시각화 (인더리디!!)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            bgcolor='white',
            aspectmode='data',
            xaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
            yaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
            zaxis=dict(backgroundcolor='white', gridcolor='lightgray')
        ),
        width=1200,
        height=800
    )
    
    return fig


def generate_3d_visualization_html(patient_id, output_path=None, visualization_type='voxel'):
    """
    환자의 이미지를 사용하여 3D 시각화 HTML 생성
    
    Args:
        patient_id: 환자 ID
        output_path: HTML 파일 저장 경로 (None이면 임시 파일)
        visualization_type: 'voxel' (복셀 기반) 또는 'mesh' (메쉬 기반) 또는 'slices' (슬라이스 탐색기)
    
    Returns:
        html_content: HTML 문자열 또는 None (실패 시)
    """
    try:
        # 환자 이미지 로드
        volume, mask_volume, spacing, image_info = load_patient_images(patient_id)
        
        if volume is None:
            logger.error(f"환자 {patient_id}의 이미지를 로드할 수 없습니다.")
            return None
        
        logger.info(f"볼륨 로드 완료: {volume.shape}")
        
        # 시각화 타입에 따라 생성
        if visualization_type == 'voxel':
            # 복셀 기반 시각화 (점 마커)
            if go is None:
                logger.error("plotly가 설치되지 않았습니다.")
                return None
            fig = visualize_3d_volume_plotly_voxel(
                volume, mask_volume, spacing,
                downsample_factor=2,
                tissue_threshold_percentile=20
            )
        elif visualization_type == 'mesh':
            # 메쉬 기반 시각화
            if go is None or measure is None:
                logger.error("plotly 또는 scikit-image가 설치되지 않았습니다.")
                return None
            fig = visualize_3d_volume_plotly(volume, spacing, mask_volume=mask_volume, threshold=0.3)
        elif visualization_type == 'slices':
            # 슬라이스 탐색기
            if go is None or make_subplots is None:
                logger.error("plotly가 설치되지 않았습니다.")
                return None
            fig = visualize_3d_volume_interactive_slices(volume, spacing)
        else:
            logger.error(f"지원하지 않는 시각화 타입: {visualization_type}")
            return None
        
        if fig is None:
            logger.error("시각화 생성 실패")
            return None
        
        # HTML 생성
        if output_path:
            fig.write_html(output_path)
            logger.info(f"HTML 파일 저장: {output_path}")
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            html_content = fig.to_html(include_plotlyjs='cdn')
        
        return html_content
        
    except Exception as e:
        logger.error(f"3D 시각화 생성 중 오류: {str(e)}", exc_info=True)
        return None


# 메인 실행 (테스트용)
if __name__ == "__main__":
    import django
    import sys
    
    # Django 설정 로드
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eventeye.settings')
    django.setup()
    
    # 테스트용 환자 ID
    patient_id = '1'  # 실제 환자 ID로 변경
    
    print("=" * 60)
    print("3D 시각화 생성 중...")
    print(f"환자 ID: {patient_id}")
    print("=" * 60)
    
    # 복셀 기반 시각화 생성
    html_content = generate_3d_visualization_html(patient_id, visualization_type='voxel')
    
    if html_content:
        output_file = f'patient_{patient_id}_3d_visualization.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n[3D 시각화 생성 완료: {output_file}]")
    else:
        print("\n[3D 시각화 생성 실패]")

