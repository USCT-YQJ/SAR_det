import os
import cv2
import json
from tqdm import tqdm


def photo():
    path = '/ssd/wqj/project/SAR/ReDet-master/work_dirs/ReDet_re50_anchor_v6_jf/Task1_results_nms/Task1_ship.txt'
    img_path = '/ssd/wqj/project/SAR/ReDet-master/data/dota15/final_test/'
    save_path = './z_final_double_head_anchor_v6_mosic_512/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_txt_path = './SAR_txt_double_head_anchor_v6_mosic_512_00/'
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    with open(path, 'r') as fr:
        labelList = fr.readlines()
        img = None
        img_name = ''
        ships = []
        for label in labelList:
            label = label.strip().split()
            name = label[0]
            score = float(label[1])
            score = float('%.2f' % score)
            if score>0.0:
                points = []
                for i in range(2, 9, 2):
                    x = int(float(label[i]))
                    y = int(float(label[i+1]))
                    if x > 512:
                        x = 512
                    if y > 512:
                        y = 512
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    points.append([x, y])
                if not name == img_name:
                    if not img_name == '':
                        cv2.imwrite(save_path+img_name+".jpg", img)
                        print(save_path+img_name+".jpg")
                        Note=open(save_txt_path+img_name+'.txt',mode='w')
                        Note.writelines(ships)
                        ships = []
                        Note.close()
                    img = cv2.imread(img_path+name+".jpg")
                    img_name = name
                ship = ''
                for i in range(4):
                    if i == 3:
                        cv2.line(img, points[i], points[0], (0, 255, 0), 2)
                    else:
                        cv2.line(img, points[i], points[i+1], (0, 255, 0), 2)
                    ship = ship+str(points[i][0])+' '+str(points[i][1])+' '
                ship = ship + 'ship ' + str(score) +'\n'
                ships.append(ship)
                cv2.putText(img, str(score), points[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
                    

if __name__ == "__main__":
    photo()