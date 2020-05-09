from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import os

from face_detect.faceDetect import cropFace2

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


        self.face_list = cropFace2(self.pull_filepath)
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


class Relight():
    def __init__(self, source, light, dest):
        self.relighting(source, light, dest)

    def preprocess_image(self, img):
        row, col, _ = img.shape
        src_img = cv2.resize(img, (256, 256))
        Lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]
        inputL = Variable(torch.from_numpy(inputL))

        return inputL, row, col, Lab

    def relighting(self, source, light, dest):
        # load model
        my_network = HourglassNet()

        my_network.load_state_dict(torch.load('trained_models/trained.pt', map_location=torch.device('cpu')))

        my_network.train(False)

        # saveFolder = os.path.join(saveFolder, source_path.split(".")[0])

        light_img, _, _, _ = self.preprocess_image(light)

        sh = torch.zeros((1, 9, 1, 1))

        _, outputSH = my_network(light_img, sh, 0)

        src_img, row, col, Lab = self.preprocess_image(source)

        outputImg, _ = my_network(src_img, outputSH, 0)

        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1, 2, 0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg * 255.0).astype(np.uint8)
        Lab[:, :, 0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))

        cv2.imwrite(os.path.join(dest,
                                 'relit.jpg'), resultLab)

ImportImg()