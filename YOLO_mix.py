import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

from collections import Counter
import config
from line import send_alert_all

N_MAX_SAMPLES = 30

def add_new_trace(df, trace):
  df.loc[len(df)] = trace # Use len(df) to append a new row
  df.to_csv('traces.csv', index=False, encoding='utf-8-sig') # 將 DataFrame 儲存為 CSV
  return df

def is_inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return X1 <= x1 and Y1 <= y1 and X2 >= x2 and Y2 >= y2

def intersection_over_area(box1, box2):
    """計算 box1 和 box2 的交集面積佔 box1 總面積的比例"""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return intersection_area / box1_area if box1_area > 0 else 0



def yolo_mix(stop_event,
             video_path=config.VIDEO,
             output_dir=config.OUTPUT,
             app_instance=None
            ):

    # 定義每個類別的顏色 (BGR 格式)
    CLASS_COLORS = {
        config.DRIVER_NAME: (255, 0, 0),      # 藍色
        config.HELMET_NAME: (0, 255, 0),      # 綠色
        config.NO_HELMET_NAME: (0, 0, 255),   # 紅色
        config.LIC_NAME: (0, 255, 255)        # 黃色
    }

    if app_instance is None:
        print("錯誤：未提供 app_instance 給 yolo_mix 函式。")
        return

    cap = cv2.VideoCapture(video_path)
    

    # 讀取現有的 CSV 檔案，如果不存在則建立一個新的
    df = pd.read_csv(config.CSV)

    # 建立一個集合，用來儲存本次執行中已經處理過的 track_id
    processed_track_ids = set()

    # 儲存潛在違規者的資訊，用於車牌投票和圖片儲存
    # {track_id: {'plates': [plate1, plate2, ...], 'best_driver_crop': img, 'best_lic_crop': img, 'best_lic_conf': 0, 'has_no_helmet': False, 'last_seen_frame': frame_num}}
    violators_tracking = {} 
    
    frame_count = 0

    # 收尾用，如果為True，把剩餘的車牌都處理完後結束
    finalize = False

    while True:
        if stop_event.is_set():
            break

        frame_count += 1

        ret, frame = cap.read()
        if not ret:
            finalize = True
            frame = np.zeros((1,1,3), np.uint8)

        # 建立一個副本用於繪製，保持原始 frame 的乾淨以供裁切
        display_frame = frame.copy()

        with app_instance.model_lock: # 鎖定以安全地讀取模型
            model = app_instance.yolo_model
        
        results = model.track(frame, persist=True, conf=config.CONF, iou=config.IOU, agnostic_nms=True, tracker=config.TRACKER_TYPE, verbose=config.VERBOSE)[0]

        current_frame_track_ids = set() # 儲存當前幀中出現的所有 track_id

        # 如果沒有偵測到任何物件或沒有追蹤ID，則跳到下一幀
        if results.boxes.id is None and not finalize:
            yield display_frame, False
            continue

        names = model.names
        boxes = results.boxes.cpu().numpy()

        # --- 新增：繪製所有偵測框與信心分數 ---
        for box in boxes:
            if box.id is None: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0])
            current_frame_track_ids.add(track_id)
            class_name = names[cls_id]
            
            # 從字典中根據類別名稱取得顏色，如果找不到則預設為白色
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            label = f'ID:{track_id} {class_name} {conf:.2f}'
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 過濾低信心分數的結果
        high_conf_indices = boxes.conf >= config.CONF
        high_conf_boxes = boxes[high_conf_indices]

        for box_big in high_conf_boxes:
            class_big = names[int(box_big.cls[0])]
            if class_big != config.DRIVER_NAME:
                continue

            # 獲取追蹤 ID
            track_id = int(box_big.id[0])
            box_big_xyxy = box_big.xyxy[0]

            # 如果是新的潛在違規者，初始化其追蹤資料
            if track_id not in violators_tracking:
                violators_tracking[track_id] = {'plates': [], 'best_driver_crop': None, 'best_lic_crop': None, 'best_lic_conf': 0, 'has_no_helmet': False, 'last_seen_frame': 0}

            violators_tracking[track_id]['last_seen_frame'] = frame_count

            # 檢查此 driver 框內是否有 no_helmet
            has_no_helmet = False
            for box_inner in high_conf_boxes:
                class_inner = names[int(box_inner.cls[0])]
                if class_inner == config.NO_HELMET_NAME:
                    box_inner_xyxy = box_inner.xyxy[0]
                    # 檢查 no_helmet 和 driver 是否有高重疊度，避免誤判路人
                    if intersection_over_area(box_inner_xyxy, box_big_xyxy) > 0.8:
                        violators_tracking[track_id]['has_no_helmet'] = True
                        has_no_helmet = True
                        break # 找到一個就夠了
            
            # 如果此 driver 內沒有偵測到 no_helmet，就跳過
            if not violators_tracking[track_id]['has_no_helmet']:
                continue

            # --- 如果確定有 no_helmet，才開始找車牌 ---
            x1b, y1b, x2b, y2b = map(int, box_big_xyxy)
            current_driver_crop = frame[y1b:y2b, x1b:x2b]

            # 更新最大張的駕駛員截圖 (這部分可以保留，但我們主要依賴車牌信心度來存圖)
            if violators_tracking[track_id]['best_driver_crop'] is None or \
               (current_driver_crop.shape[0] * current_driver_crop.shape[1] > \
                violators_tracking[track_id]['best_driver_crop'].shape[0] * violators_tracking[track_id]['best_driver_crop'].shape[1]):
                violators_tracking[track_id]['best_driver_crop'] = current_driver_crop

            # 在確定違規的 driver 框內尋找車牌
            for box_small in high_conf_boxes:
                class_small = names[int(box_small.cls[0])]
                if class_small != config.LIC_NAME:
                    continue
                
                box_small_xyxy = box_small.xyxy[0]
                # 1. 檢查車牌是否在騎士框內
                if not is_inside(box_small_xyxy, box_big_xyxy):
                    continue
                
                # 2. 新增判斷：檢查車牌是否在騎士框的垂直中心線下方
                '''center_y_big = y1b + (y2b - y1b) / 2
                y1s, _, _, _ = map(int, box_small_xyxy)
                if y1s < center_y_big:
                    continue'''
                
                #------以下處理為確認違規後的操作------                
                x1s, y1s, x2s, y2s = map(int, box_small_xyxy)
                lic_crop = frame[y1s:y2s, x1s:x2s]
                
                # 進行 OCR 並收集車牌號碼
                plate_results = config.LPR.run(lic_crop)
                if plate_results:
                    plate_text = plate_results[0].strip('_')
                    # 只有在車牌號碼有效，且收集到的樣本數小於 10 時，才進行收集
                    if plate_text and len(violators_tracking[track_id]['plates']) < N_MAX_SAMPLES:
                        violators_tracking[track_id]['plates'].append(plate_text)
                        print(f"Track ID {track_id}: 收集到車牌 '{plate_text}'。目前列表: {violators_tracking[track_id]['plates']}")

                        # 如果當前車牌的偵測信心度更高，就更新最佳截圖
                        current_lic_conf = box_small.conf[0]
                        if current_lic_conf > violators_tracking[track_id]['best_lic_conf']:
                            violators_tracking[track_id]['best_lic_conf'] = current_lic_conf
                            violators_tracking[track_id]['best_driver_crop'] = current_driver_crop
                            violators_tracking[track_id]['best_lic_crop'] = lic_crop
                            print(f"Track ID {track_id}: 更新最佳截圖，信心度: {current_lic_conf:.2f}")

                # 找到一個符合條件的車牌後，就假設它是這個 driver 的車牌，
                # 跳出尋找車牌的迴圈，繼續處理下一個 driver。
                break

        # --- 幀處理結束後：檢查並處理已完成追蹤的違規者 ---
        track_ids_to_finalize = []
        for track_id, data in list(violators_tracking.items()):
            # 檢查是否符合處理條件：1. 未處理過 2. 確定未戴安全帽 3. 已收集到車牌
            if track_id not in processed_track_ids and data['has_no_helmet'] and data['plates']:
                # 觸發投票的條件：
                # 1. 物件已離開畫面 (不在當前幀出現)
                # 2. 或已收集到足夠數量的車牌 (10 個)
                #object_left_frame = (track_id not in current_frame_track_ids)
                print(len(data['plates']))
                sufficient_plates_collected = (len(data['plates']) >= N_MAX_SAMPLES)

                if sufficient_plates_collected: 
                    track_ids_to_finalize.append(track_id)

        if finalize:
            track_ids_to_finalize = list(violators_tracking.keys())
            print(f"track_ids_to_finalize: {track_ids_to_finalize}")

        if len(track_ids_to_finalize) > 0:
            finalize_print(df)

        def finalize_print(df):
            for track_id in track_ids_to_finalize:
                data = violators_tracking[track_id]
                
                # 進行投票
                plate_counts = Counter(data['plates'])
                most_common_plate = plate_counts.most_common(1)[0][0] if plate_counts else ""
                print(f"Track ID {track_id}: 投票結束！最終勝利的車牌號碼是: {most_common_plate}")

                # 只有在成功投票選出車牌，且該車牌尚未記錄在 CSV 中時才處理
                if most_common_plate and most_common_plate not in df['kar_id'].values:
                    # --- 存檔與警報 ---
                    if data['best_driver_crop'] is not None and data['best_lic_crop'] is not None:
                        # 儲存包含駕駛和車牌的大圖 (driver crop)
                        big_path = os.path.join(output_dir, f"trackID_{track_id}_{config.DRIVER_NAME}.jpg")
                        cv2.imwrite(big_path, data['best_driver_crop'])
                        print("[Saved big]", big_path)

                        # 儲存車牌的小圖 (lic crop)
                        small_path = os.path.join(output_dir, f"trackID_{track_id}_{config.LIC_NAME}.jpg")
                        cv2.imwrite(small_path, data['best_lic_crop'])
                        print("[Saved small]", small_path)

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_trace = {'video_name': config.FILENAME, 'kar_id': most_common_plate, 'date': current_time, 'helmet': 'False', 'image_url': big_path}
                        
                        if app_instance:
                            df = app_instance.add_new_trace_and_update(new_trace)
                        else:
                            df = add_new_trace(df, new_trace)

                        message_to_send = (f"⚠️系統偵測到違規⚠️\n"
                                        f"違規項目: 未戴安全帽\n"
                                        f"車牌號碼: {new_trace['kar_id']}\n"
                                        f"影片檔名: {os.path.basename(config.FILENAME)}\n"
                                        f"影片時間: {new_trace['date']}")
                        #send_alert_all(message_to_send)

                        processed_track_ids.add(track_id)
                
                # 處理完畢後，從追蹤字典中移除，避免重複處理
                del violators_tracking[track_id]
                print(violators_tracking)

        # 回傳繪製好偵測框的畫面
        yield display_frame, True # 有偵測到物件，但可能不是新違規

        if finalize:
            break

    # End of while loop
    cap.release()