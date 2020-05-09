from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import os

from faceDetect import cropFace
from relight import Relight

class ImportImg():

    def __init__(self):
        self.root = Tk()
        self.root.title('Face Relighting - Import Image') 

        self.pull_button = Button(self.root, text="Select Image To Pull Lighting From", command=self.getPullFile, padx=50, pady=50)
        self.apply_button = Button(self.root, text="Select Image To Apply Lighting To", command=self.getApplyFile, padx=50, pady=50)

        self.next_button = Button(self.root, text="Next", state=DISABLED, command=self.loadNextSection)

        self.next_button.grid(row=2, column=1)
        self.pull_button.grid(row=0, column=0)
        self.apply_button.grid(row=0, column=2)

        self.pull_filepath = None
        self.apply_filepath = None

        self.root.mainloop()


    def getPullFile(self):
        self.pull_filepath = filedialog.askopenfilename(title="Select Image", filetypes=(("png", ".png"), ("jpeg", ".jpeg"), ("jpg", ".jpg")))
        pull_label = Label(self.root, text=self.pull_filepath, bg="blue", fg="white")
        pull_label.grid(row=1, column=0)

        if self.apply_filepath != None:
            self.nextButtonReady()


    def getApplyFile(self):
        self.apply_filepath = filedialog.askopenfilename(title="Select Image", filetypes=(("png", ".png"), ("jpeg", ".jpeg"), ("jpg", ".jpg")))
        apply_label = Label(self.root, text=self.apply_filepath, bg="blue", fg="white")
        apply_label.grid(row=1, column=2)

        if self.pull_filepath != None:
            self.nextButtonReady()

    def nextButtonReady(self):
        self.next_button.config(state=NORMAL, bg="green")
    
    def loadNextSection(self):
        self.root.destroy()
        SelectLightFace(self.pull_filepath, self.apply_filepath)

class SelectLightFace():
    def __init__(self, pull_filepath, apply_filepath):
        self.root = Tk()
        # self.root.geometry("600x600")
        self.root.title('Face Relighting - Select Lighting Face') 
        
        self.pull_filepath = pull_filepath
        self.apply_filepath = apply_filepath

        instruction_label = Label(self.root, text="Pick a face on the left to use as lighting reference")

        apply_img = ImageTk.PhotoImage(Image.open(self.apply_filepath).resize((512, 512)))
        apply_img_label = Label(self.root, image=apply_img)


        self.face_list = cropFace(self.pull_filepath)
        self.face_list_np = self.face_list.copy()
        self.convertNumpyImgs(self.face_list)
        self.curr = 0

        self.curr_carosel = Label(self.root, image=self.face_list[0])

        left_button = Button(self.root, text="<<", command=self.left)
        right_button = Button(self.root, text=">>", command=self.right)
        select_button = Button(self.root, text="Apply Lighting", command=self.select)


        instruction_label.grid(row=0, column=1)
        left_button.grid(row=1, column=0)
        self.curr_carosel.grid(row=1, column=1)
        right_button.grid(row=1, column=2)
        apply_img_label.grid(row=1, column=3)
        select_button.grid(row=2, column=1)

        self.root.mainloop()

    def left(self):
        if self.curr > 0:
            self.curr -= 1
            self.curr_carosel.config(image=self.face_list[self.curr])
    
    def right(self):
        if self.curr < (len(self.face_list) - 1):
            self.curr += 1
            self.curr_carosel.config(image=self.face_list[self.curr])

    def select(self):
        
        pull_img = self.face_list_np[self.curr]
        
        apply_img = cv2.imread(self.apply_filepath)

        dest = filedialog.askdirectory(title="Select Output Folder")

        Relight(apply_img, pull_img, dest)

        self.root.destroy()

    def convertNumpyImgs(self, np_faces):
        for i in range(len(np_faces)):
            np_faces[i] = ImageTk.PhotoImage(Image.fromarray(np_faces[i], 'RGB').resize((512, 512)))


ImportImg()