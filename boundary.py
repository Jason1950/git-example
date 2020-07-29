import cv2
import numpy as np
import time
import sys
import math

class Boundary(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.start_time = time.time()
        self.fps = 0
        self.index = 0
        self.gamma = 3.5
        self.process_resize = 0.5
        self.video_read(self.file_name)

    def gamma_trans(self, img, gamma):  # gamma函数处理
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
        return cv2.LUT(img, gamma_table) 
    
    def video_read(self, file_video):
        self.cap = cv2.VideoCapture(file_video)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f'Video  FPS : {round(fps,1)} , Width : {self.width} , Height : {self.height} ')
        # save the video
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # 使用 XVID 編碼
        self.out = cv2.VideoWriter(f'output_cnts.mp4'
                                    , fourcc
                                    , 30.0
                                    ,(int(self.width*self.process_resize) 
                                    ,int(self.height*self.process_resize)))
        # self.out = cv2.VideoWriter(f'output_cnts.avi', fourcc, 30.0, (800,800))
    
    def J_conts(self, frame, color, line_width=1, low=50, hight=150):
        lap_temp = cv2.Laplacian(frame.copy(), cv2.CV_16S, ksize=1)
        dst = cv2.convertScaleAbs(lap_temp) # 轉回uint8
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        _ , dst_gray = cv2.threshold(dst_gray, low, hight, 0)
        cnts, _ = cv2.findContours(dst_gray,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cnts, -1, color, line_width)
        return frame, cnts

    def J_fps(self):
        if time.time() - self.start_time > 1:
            self.fps = self.index
            self.start_time = time.time()
            self.index = 0

    def gamma_trans(self, img, gamma):  # gamma函数处理
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
        return cv2.LUT(img, gamma_table) 

    def feature_gamma_v2(self, img, gamma_val):
        image_gamma_correct = self.gamma_trans(img, gamma_val)   # gamma变换
        return image_gamma_correct

    def light_percent(self, frame, limit=200, show=True):
        sum_temp = np.sum(frame>limit)
        if show:
            print(f'[Info -- light_percent --] light_{limit} sum : ',sum_temp)
            print(f'[Info -- light_percent --] frame shape : ',frame.shape)
        ta = 1
        for i in range(frame.shape[-1]):
            ta *= frame.shape[i]  
        if show:
            print(f'[Info -- light_percent --] frame total pixel : ',ta)
        light_percent = round(sum_temp/ta*100,1)
        state = False
        if light_percent > 15 :
            state = True
        return light_percent, state  #float

    def all_auto_exposure_opt(self, frame, light_limit): #, fp, record=False):
        light1, state = self.light_percent(frame.copy(), limit=light_limit, show=False)
        light2 = None
        gamma = 1.0
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(frame_gray)
        if state:
            if light1 > 30 and mean > 0:
                gamma = math.log10(0.5)/math.log10(mean/255)
                gamma -= 0.4
                frame = self.feature_gamma_v2( frame, round(gamma,1))

            light2, state = self.light_percent(frame, limit=light_limit, show=False)
        # if record:
        #     ## write the litht distribution
        #     fp.write(str(light1))
        #     fp.write(', ')
        #     fp.write(str(gamma))
        #     fp.write(', ')
        #     fp.write(str(light2))
        #     fp.write('\n')

        return frame

    def re2(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(frame_gray)
        gamma = math.log10(0.5)/math.log10(mean/255)
        frame = self.feature_gamma_v2( frame, round(gamma,1))
        return frame

    def main(self, rate):
        j = 0
        while True:
            ret, frame = self.cap.read()
            if ret:

                self.J_fps()

                frame = cv2.resize(frame, ( int(frame.shape[1]*self.process_resize) ,int(frame.shape[0]*self.process_resize) ))
                
                ## original frame
                # frame2 = frame.copy()

                ## dark gamma = 2.5
                # frame_gamma_25 = self.gamma_trans(frame2, 2.5)
  
                ## light gamma = 0.8
                # frame_gamma_08 = self.gamma_trans(frame2, 0.8)

                ## Calculation the frame's boundary 
                # frame_white, cnts_w = self.J_conts( frame.copy(), color=(255,255,255), line_width=1, low=40, hight=60)
                
                

                frame_black, cnts_b = self.J_conts( frame.copy(), color=(0,255,0), line_width=2, low=40, hight=60)
                
                
                resized = frame.copy()
                frame = frame_black
                for (i, c) in enumerate(cnts_b):
                    j+= 1
                    (x, y, w, h) = cv2.boundingRect(c)
                    if w <10 or h < 10 or (w>30 and h>30):
                        continue
                    print("Coin #{}".format(i + 1))
                    coin = resized[y:y + h, x:x + w]
                    
                    # coin = self.all_auto_exposure_opt(coin, 200)
                    coin = self.re2(coin)
                    cv2.imshow("Coin", coin)
                    frame[y:y + h, x:x + w] = coin
                    '''mask = np.zeros(resized.shape[:2], dtype = "uint8")
                    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
                    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
                    mask = mask[y:y + h, x:x + w]
                    coin_show = cv2.bitwise_and(coin, coin, mask = mask)'''
                    # cv2.imshow("Masked Coin", coin_show)
                    # cv2.imwrite(f'./../trash_boundary/Masked_image{j}.png',coin_show)

                ## Update the frame with new boundary
                ## Compare pair
                # frame[:,:int(frame.shape[1]/2)] = frame_white[:,:int(frame.shape[1]/2)]
                # frame[:,int(frame.shape[1]/2):] = frame_black[:,int(frame.shape[1]/2):]

                ## Full cnts 
                frame = frame_black

                ## Show and Save vidoe 
                # self.out.write(frame)
                frame = cv2.resize(frame, ( int(frame.shape[1]*rate) ,int(frame.shape[0]*rate) ))
                cv2.imshow('laplacian', frame)
                cv2.waitKey(1)
                
                ## Show Info ! 
                sys.stdout.write("\r{0}".format(f'360 video fps : {self.fps}'))
                sys.stdout.flush()

                self.index +=1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # Boundary('./../360LocationMovie/'+'360MotionCrossWall_D02.mp4').main(rate = 0.3)
    # Boundary('./../ICL_360/'+'2020-07-23_18-12-21.mov').main(rate=1.0)
    # Boundary('./../tmc/'+'tmc30_gamma2.0.avi').main(rate=0.6)
    Boundary('./../360LocationMovie/'+'360Motion_B.mp4').main(rate = 0.6)
    # Boundary('./'+'output_cnts_dspa_move_wall.mp4').main(rate = 0.6)
    # Boundary('./../360LocationMovie/'+'360E_Jason_0708.avi').main(rate = 0.3)
