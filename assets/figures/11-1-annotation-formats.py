"""
Annotation Format Comparison: COCO vs VOC vs YOLO
=================================================
Visualizes the same image's annotations in three different formats.

Output: assets/figures/11-1-annotation-formats.png
"""
import matplotlib.pyplot as plt
import json
import numpy as np


np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('目标检测标注格式对比: COCO vs VOC vs YOLO',
             fontsize=15, fontweight='bold', y=0.98)


# Mock image with 2 objects
img_w, img_h = 500, 400
objects = [
    {'name': 'cat',   'bbox_xywh': [50, 60, 180, 220], 'id': 1},   # x, y, w, h
    {'name': 'dog',   'bbox_xywh': [280, 100, 150, 180], 'id': 2},
]

# === Panel 1: Visualization ===
ax = axes[0]
ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)
ax.set_aspect('auto'); ax.set_title('Sample Image with 2 BBoxes',
                                    fontsize=12, fontweight='bold')
ax.add_patch(plt.Rectangle((0, 0), img_w, img_h, fill=True, facecolor='#F5F5F5'))
# Draw fake objects (just colored rectangles for illustration)
for obj in objects:
    x, y, w, h = obj['bbox_xywh']
    if obj['name'] == 'cat':
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor='#FFB6C1', alpha=0.7))
    else:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor='#87CEEB', alpha=0.7))
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2))
    ax.text(x, y - 15, obj['name'], fontsize=10, color='green', fontweight='bold')
ax.set_xticks([0, 250, 500])
ax.set_yticks([0, 200, 400])
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')


# === Panel 2: COCO JSON ===
ax = axes[1]
ax.axis('off')
ax.set_title('COCO JSON Format\n(工业标准, 推荐)', fontsize=12, fontweight='bold')

coco_example = {
    "info": {"description": "Demo"},
    "images": [{"id": 1, "width": 500, "height": 400, "file_name": "img1.jpg"}],
    "categories": [
        {"id": 1, "name": "cat"},
        {"id": 2, "name": "dog"}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1,
         "bbox": [50, 60, 180, 220], "area": 39600, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2,
         "bbox": [280, 100, 150, 180], "area": 27000, "iscrowd": 0}
    ]
}
coco_text = json.dumps(coco_example, indent=2, ensure_ascii=False)
ax.text(0.02, 0.95, coco_text, fontsize=7.5, family='monospace', va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#E8F4FD', edgecolor='#1F77B4'))


# === Panel 3: VOC XML vs YOLO TXT ===
ax = axes[2]
ax.axis('off')
ax.set_title('VOC XML vs YOLO TXT', fontsize=12, fontweight='bold')

voc_xml = '''<annotation>
  <folder>images</folder>
  <filename>img1.jpg</filename>
  <size><width>500</width>
        <height>400</height></size>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>50</xmin>  <ymin>60</ymin>
      <xmax>230</xmax> <ymax>280</ymax>
    </bndbox>
  </object>
  <object>
    <name>dog</name>
    <bndbox>
      <xmin>280</xmin> <ymin>100</ymin>
      <xmax>430</xmax> <ymax>280</ymax>
    </bndbox>
  </object>
</annotation>'''

yolo_txt = '''# img1.jpg 500x400
# class_id  x_center  y_center  w       h  (normalized 0-1)
1            0.280     0.425    0.360   0.550
2            0.710     0.475    0.300   0.450

# class names (in classes.txt):
# 0: person
# 1: cat
# 2: dog'''

comparison = f"""━━━ VOC XML (左上角坐标 + 宽高) ━━━
{voc_xml}

━━━ YOLO TXT (中心点 + 宽高, 归一化) ━━━
{yolo_txt}
"""
ax.text(0.02, 0.95, comparison, fontsize=7, family='monospace', va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='black'))


plt.tight_layout()
plt.savefig('assets/figures/11-1-annotation-formats.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/11-1-annotation-formats.png')
