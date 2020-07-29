import cv2
import numpy as np
#import win32api as win32
class BaseBallSpeed(object):
    def __init__(self, file_name, pos, base, state, resize):
        #初始化 - 攝影機 / 影片
        self.file_name = file_name
        self.cap = cv2.VideoCapture(self.file_name)
        self.resize = resize
        self.base = base
        self.flip_state = state

        # 設定影像尺寸
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print( self.width, self.height)

        # 設定擷取影像的尺寸大小
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 顯示影片格式之 FPS 
        print(f'\nThe video {self.file_name} fps is ',self.cap_fps)
        print(f'\nEvery frame duration time is {round((1/self.cap_fps),3)} sec \n')

        # 計算畫面面積
        self.area = self.width * self.height
        
        # other parameter
        self.thresh_old = np.array([0,0])
        self.t_state = False
        self.click_state = True
        self.detect_ball_state = False
        self.time = 0
        self.run_count = 0
        self.click = 0
        self.click_pos = np.empty([0,2])

        self.meter_rate = 1
        # some
        self.detect_number = 0
        # count frame not only black times 
        self.frame_count = 0
        self.pos_x = pos[0]
        self.pos_w = pos[1]
        self.pos_y = pos[2]
        self.pos_h = pos[3]

    def run(self, low, high):
        # 初始化平均影像
        ret, frame = self.cap.read()
        avg = cv2.blur(frame, (4, 4))
        avg_float = np.float32(avg)
        while(self.cap.isOpened()):
            # 讀取一幅影格
            ret, frame = self.cap.read()

            # 若讀取至影片結尾，則跳出
            if ret == False:
                break
            if self.flip_state:
                frame = cv2.flip(frame, -1)
            # 模糊處理 ，模糊程度: 低
            blur = cv2.blur(frame, (4, 4))

            # 計算目前影格與平均影像的差異值 # this array just like frame array 
            diff = cv2.absdiff(avg, blur)

            # 將圖片轉為灰階
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # 篩選出變動程度大於門檻值的區域
            # [80,255] is great threshold to filter other item but baseball
            ret, thresh = cv2.threshold(gray[self.pos_y:(self.pos_y+self.pos_h)
                ,self.pos_x:(self.pos_x+self.pos_w+12)], low, high, cv2.THRESH_BINARY)
            #low : 80 , high : 255
            
            # =======================
            #    觀察thresh 的狀態
            # =======================
            cv2.imshow('thresh', thresh)

            if self.thresh_old.any() == 0 and thresh.any() != 0:
                self.t_state = True
                self.frame_count = 0
                self.detect_number += 1

                #print('thresh.any() : ', thresh.any()==0, ' , thresh_old.any() : ',self.thresh_old.any() ==0)
                if self.detect_ball_state :
                    print('correct ball entry (O) !!')  
                else:                
                    print('error entry (x)')
                    
            elif self.thresh_old.any() != 0 and thresh.any() == 0:
                if self.detect_ball_state :
                    print('correct ball exit (O) !!')
                    acc = round((self.pos_w * self.meter_rate *3.6 / self.time ),1)
                    print(f'The {self.detect_number} times detect the ball , duration : ', 
                            round(self.time, 3), 
                            ' sec , v : ',acc ,
                            ' km/hr, frame count : ', 
                            self.frame_count, 
                            ' , w : ', 
                            self.pos_w,'\n')
                else:
                    print('error exit (x)')
                self.t_state = False
                self.detect_ball_state = False


            if self.t_state and thresh.any() != 0:
                self.frame_count += 1
                self.time = (self.frame_count / self.cap_fps)
                #print('detect the ball in the region ! , times : ', self.time, ' sec , frame count : ', self.frame_count)
                  


            self.thresh_old = thresh.copy()
            
            # 使用型態轉換函數去除雜訊
            # kernel = np.ones((5, 5), np.uint8)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            # cv2.imshow('thresh',thresh)

            # 產生等高線
            # cntImg, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # if cnts is not None:
            i = 0
            # print('c len',len(cnts))
            for c in cnts:
                # cnts 會持續疊加，有幾個frame偵測到物件 ctns就有幾個
                # 且最後更新之x,y 為最後一組ctns!!!
                #print('c ',i) 
                # 忽略太小的區域
                #if cv2.contourArea(c) < 25:
                #    continue

                # 計算等高線的外框範圍
                (x, y, w, h) = cv2.boundingRect(c)
                x = x + self.pos_x #200
                y = y + self.pos_y #340

                #if (abs(x - self.pos_x) < 3 or abs(x - (self.pos_w+self.pos_x))<3):
                if abs(x - (self.pos_w+self.pos_x)) < 10 :
                    self.detect_ball_state = True
                i += 1
                # 畫出外框
                if self.detect_ball_state:
                    cv2.rectangle(frame, (x-5, y-5), (x + w+5, y + h+5), (0, 255, 0), 2)
            

 

            
            cv2.rectangle(frame, (self.pos_x, self.pos_y), ((self.pos_x+self.pos_w), (self.pos_y + self.pos_h)), (255, 100, 0), 3)
            # 200,340  700,520
            # [340:520,200:700]
            # 畫出等高線（除錯用）
            #cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
            if self.run_count > 2:
                # print('line???',self.click_pos[0][0],self.click_pos[0][1],self.click_pos[1][0],self.click_pos[1][1])
                # print('count > 2 , click position :',self.click_pos)
                x1, y1 = int(self.click_pos[0][0]*100/self.resize), int(self.click_pos[0][1]*100/self.resize)
                x2, y2 = int(self.click_pos[1][0]*100/self.resize), int(self.click_pos[1][1]*100/self.resize)
                # cv2.line(frame,(self.click_pos[0][0],self.click_pos[0][1]),
                # (self.click_pos[1][0],self.click_pos[1][1]),(0,100,255),1)
                cv2.line(frame,(x1, y1),(x2, y2),(0,100,255),1)
                # self.meter_rate = self.base /((self.click_pos[0][0] - self.click_pos[1][0])**2 + (self.click_pos[0][1]-self.click_pos[1][1])**2 )**0.5
                self.meter_rate = self.base /((x1 - x2)**2 + (y1-y2)**2 )**0.5
            
            # 顯示偵測結果影像
            # cv2.imshow('frame', frame)
                    
            # frame = cv2.resize(frame,(600,600))
            # cv2.imshow('frame',frame)

            frame2 = cv2.resize(frame, (int(self.width*self.resize/100) , int(self.height*self.resize/100)))      
            cv2.imshow(f'frame resize : {self.resize} % ',frame2)

            #cv2.imshow('frame2', frame[340:520,200:700]) #[y,x]
            def onmouse(event, x, y, flags, param):   #標準滑鼠互動函式
                # if event==cv2.EVENT_MOUSEMOVE:      #當滑鼠移動時
                if event == cv2.EVENT_LBUTTONDOWN and (not self.click_state):
                    print('x,y :', x,y)
                    self.click_pos = np.vstack([self.click_pos,(x,y)])
                    self.click += 1
                    print('== ', self.click,self.click_state)
                    if self.click > 1:
                        print('click ok!!')
                        self.click_state = True
                        self.click_pos = self.click_pos.astype(int)
                        print('click array : ',self.click_pos)
                        click_dis = ((self.click_pos[0][0] - self.click_pos[1][0])**2 + (self.click_pos[0][1]-self.click_pos[1][1])**2 )**0.5
                        print('click distance : ' , click_dis , ' , dis zoom rate : ', self.base/click_dis )

            if self.run_count == 2:
                self.click_state = False
                # cv2.setMouseCallback("frame", onmouse)   #回撥繫結視窗 
                cv2.setMouseCallback(f'frame resize : {self.resize} % ', onmouse)   #回撥繫結視窗 
                while True:
                    cv2.waitKey(10)
                    if self.click_state:
                        break
                    
            ## ======================================
            ##        h e r e ! ! ! please dont forget!! 
            ## ======================================
            # cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 更新平均影像
            cv2.accumulateWeighted(blur, avg_float, 0.01)
            avg = cv2.convertScaleAbs(avg_float)
            self.run_count += 1
            # print('run_count : ',self.run_count)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ## 010 position!!
    x, w = 1050, 100
    y, h = 350, 80
    # x,w = 500 , 800
    # y,h = 300 , 120
    # y,h = 340,180
    # x,w = 200,500

    ##hdr position !!
    x2, w2 = 416, 250
    y2, h2 = 298, 180

    pos1 = [x, w, y, h]
    pos2 = [x2, w2, y2, h2]

    # BaseBallSpeed('010.mp4', pos1, 18.44, True, 80).run(45,100)
    BaseBallSpeed('021.mp4', pos1, 18.44, True, 100).run(45,100)
    # BaseBallSpeed('94hdr.mp4', pos2, 54.5, False, 70).run(30,100)

    ## BaseBallSpeed aeg : video, roi position, base line, resize percent
    ## base line : 5.45 or 18.44 , means pichting and hitting , pichting size
    ## True of False : flip -1 or not ! Because some videos will flip auto dont reason!
    ## run arg : low and high detecting