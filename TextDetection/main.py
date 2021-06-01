import LoadImg_ModelEAST
from LoadImg_ModelEAST import t
import text_detection
from text_detection import Img_proc
import text_recognization
from text_recognization import text_recog
import cv2

text_proessing = text_recog()
LoadImg_Model = t()
Img_p = Img_proc()
# Read Image
image = LoadImg_Model.loadImage()

# Processing Image
args = Img_p.args
image_copy = Img_p.orig
net =  LoadImg_Model.loadModel()
img_resize = Img_p.resizeIMG(image_copy, args)
blob = Img_p.blob(image_copy)
scores, geometry = Img_p.geomatric(net)
print('1')
(boxes, confidence_val) = Img_p.predictions(scores, geometry)
fianl_boxes = Img_p.finalBox(net)
#output = Img_p.output(image_copy, fianl_boxes)

# Processing text on image
print('2')
converted = text_proessing.converted(boxes)
results = text_proessing.results
print('3')
output_text = text_proessing.output_text(results)
print('4')
res_img = text_proessing.output_textonpic(results)
