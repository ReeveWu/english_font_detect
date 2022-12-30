import cv2
import shutil
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from dictionary import *

def win_work():
    width = 700
    height = 240

    win=tk.Tk()
    win.title("welcome")
    win.resizable(0, 0)
    win.attributes("-topmost",1)

    screen_width, screen_height = win.winfo_screenwidth(), win.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    win.geometry('%dx%d+%d+%d' % (width, height, x, y))

    text_lb=tk.Label(fg="black",text='Move to the window you want to identify\n and press the button to screenshot.')
    # text_lb.config(width=50, height=5)
    text_lb.config(font="Georgia 19")
    text_lb.config(justify="center")
    text_lb.place(x=73, y=40)

    def finish():
        win.destroy()

    Screenshot = tk.Button(text='screenshot', font="Georgia 12")
    Screenshot.config(fg="black", relief='ridge')
    Screenshot.config(width=38,height=1)
    Screenshot.config(command=finish)
    Screenshot.place(x=120,y=140)

    win.mainloop()

state = True
target_font = []

def show_predict(top_idx, TopBER):
    for index in top_idx:
        target_font.append(category2font[index])
    def show_fonts():
        load = tk.Label(fg="black", text='loading...')
        load.config(font="Georgia 12", width=39)
        load.config(justify="center")
        load.place(x=10, y=435)
        global state
        if state:
            if os.path.exists('fonts'):
                for fileName in os.listdir('fonts'):
                    os.remove('fonts/' + fileName)
                os.rmdir('fonts')

            font_list = os.listdir('project_fonts/Single')
            font_list = font_list + os.listdir('project_fonts/Diversity')
            os.mkdir('fonts')
            for name in target_font:
                for i in font_list:
                    if name in i:
                        font_name = i
                        source = r'project_fonts/Single/{0}'.format(font_name)
                        if os.path.isfile(source):
                            new_sourse = []
                            new_sourse.append(source)
                        else:
                            source = os.listdir('project_fonts/Diversity/{0}'.format(name))
                            for index, f in enumerate(source):
                                source[index] = 'project_fonts/Diversity/{0}/'.format(name) + f
                            new_sourse = source

                        for f in new_sourse:
                            destination = 'fonts/{0}'.format(f.split('/')[-1])
                            shutil.copyfile(f, destination)
            state = False
            os.system("explorer.exe %s" % r'fonts')
        else:
            os.system("explorer.exe %s" % r'fonts')
        load.destroy()

    img = cv2.imread('origin/output.png', -1)
    size = 70/img.shape[0]
    img = cv2.resize(img, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
    if img.shape[1] > 470:
        size = 470 / img.shape[1]
        img = cv2.resize(img, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
    pos = img.shape[0]
    cv2.imwrite('origin/output.png', img)

    width = 490
    height =500

    win=tk.Tk()
    win.title("Identification Results")
    win.resizable(0, 0)

    canvas = tk.Canvas(height=5)

    screen_width, screen_height = win.winfo_screenwidth(), win.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    win.geometry('%dx%d+%d+%d' % (width, height, x, y))

    img = Image.open('origin/output.png')
    tk_img = ImageTk.PhotoImage(img)
    top_distance = (80-pos)/2
    target = tk.Label(win, image=tk_img, width=400)
    target.pack(fill='x', ipady=top_distance)

    canvas.create_line(0, top_distance, 550, top_distance, width=1, fill='gray',  dash=(4, 1))
    canvas.pack(fill='x')

    text1 = tk.Label(fg="black", text='The top 3 most similar:')
    text1.config(font="Georgia 14 underline")
    text1.config(justify="left")
    text1.place(x=10,y=105)

    f1 = target_font[0]
    font1 = tk.Label(fg="black", text='1. {0}.ttf'.format(f1))
    font1.config(font="Georgia 12")
    font1.config(justify="left")
    font1.place(x=10, y=145)

    a1 = float('{0:.2f}'.format(TopBER[0]))*100.0
    acc1 = tk.Label(fg="black", text='-- {0:.2f} %'.format(a1))
    acc1.config(font="Georgia 12", width=13)
    acc1.config(justify="right")
    acc1.place(x=350, y=145)

    list_of_key = list(number2category.keys())
    list_of_value = list(number2category.values())

    i = list_of_key[list_of_value.index(top_idx[0])]
    i = str(i).zfill(2)
    img = Image.open('fonts_demo/fonts_demo-{0}.png'.format(i))
    f1_img = ImageTk.PhotoImage(img)
    f1_demo = tk.Label(win, image=f1_img)
    f1_demo.place(x=10, y=175)

    f2 = target_font[1]
    font2 = tk.Label(fg="black", text='2. {0}.ttf'.format(f2))
    font2.config(font="Georgia 12")
    font2.config(justify="left")
    font2.place(x=10, y=220)

    a2 = float('{0:.2f}'.format(TopBER[1]))*100.0
    acc2 = tk.Label(fg="black", text='-- {0:.2f} %'.format(a2))
    acc2.config(font="Georgia 12", width=13)
    acc2.config(justify="right")
    acc2.place(x=350, y=220)

    i = list_of_key[list_of_value.index(top_idx[1])]
    i = str(i).zfill(2)
    img = Image.open('fonts_demo/fonts_demo-{0}.png'.format(i))
    f2_img = ImageTk.PhotoImage(img)
    f2_demo = tk.Label(win, image=f2_img)
    f2_demo.place(x=10, y=250)

    f3 = target_font[2]
    font3 = tk.Label(fg="black", text='3. {0}.ttf'.format(f3))
    font3.config(font="Georgia 12")
    font3.config(justify="left")
    font3.place(x=10, y=295)

    a3 = float('{0:.2f}'.format(TopBER[2]))*100.0
    acc3 = tk.Label(fg="black", text='-- {0:.2f} %'.format(a3))
    acc3.config(font="Georgia 12", width=13)
    acc3.config(justify="right")
    acc3.place(x=350, y=295)

    i = list_of_key[list_of_value.index(top_idx[2])]
    i = str(i).zfill(2)
    img = Image.open('fonts_demo/fonts_demo-{0}.png'.format(i))
    f3_img = ImageTk.PhotoImage(img)
    f3_demo = tk.Label(win, image=f3_img)
    f3_demo.place(x=10, y=325)

    open = tk.Button(text='Open the fonts', font="Georgia 14")
    open.config(fg="black", relief='ridge')
    open.config(width=32)
    open.config(command=show_fonts)
    open.place(x=10, y=380)

    if not np.min(TopBER) >= 0:
        load = tk.Label(fg="red", text='Less accurate identification results.\n'
                                       '(It is possible that the texts obtained are too blurry or crowded.)')
        load.config(font="Georgia 10", width=48)
        load.config(justify="center")
        load.place(x=1, y=435)

    win.mainloop()
