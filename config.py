from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# ===== 廣域設定 =====
CSV = 'traces.csv'
CONF = 0.25                              #信心度
IOU = 0.5                               #重疊度
TRACKER_TYPE = 'botsort.yaml'           #追蹤器類型: botsort.yaml or bytetrack.yaml
VERBOSE = False                         #是否顯示詳細訊息


# ===== 影片 =====
VIDEO = None
FILENAME = None

#https://trafficvideo.ttcpb.gov.tw/e5cd69f5
# ===== 模型預設 =====
MODEL_PATH = 'models/Distillation.pt'                 #混合模型路徑
MODEL_GLOBAL = 'cct-s-v1-global-model'                          #辨識車牌模型
LPR = LicensePlateRecognizer(MODEL_GLOBAL, device='cpu') # 初始化車牌辨識器
# ===== folder設定 =====
OUTPUT = 'output'

# ===== TK介面 =====
TK_TITLE = "未戴安全帽AI自動偵測後台"     #Title
TK_GEOMETRY = "1080x680"                #介面大小
NEW_GEOMETRY = "400x700"                #視窗大小

VIDEO_WIDTH = 700                       #影片寬
VIDEO_HEIGHT = 400                      #影片高

# ===== Model分類 =====
LIC_ID = 0                              #車牌ID
DRIVER_ID = 1                           #駕駛ID
HELMET_ID = 2                           #安全帽ID
NO_HELMET_ID = 3                        #無安全帽ID

LIC_NAME = 'lic'                          #車牌名稱
DRIVER_NAME = 'driver'                  #駕駛名稱
HELMET_NAME = 'helmet'                  #安全帽名稱
NO_HELMET_NAME = 'no_helmet'            #無安全帽名稱