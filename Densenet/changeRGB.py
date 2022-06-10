from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=500,height=500):
    img=Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))

    except Exception as e:
        print(e)
    print("转换成功")
i=0
for jpgfile in glob.glob("../Protect_Animal_Data/台湾猴/*.jpeg"):  # jfif jpeg png webp
    print(jpgfile)
    convertjpg(jpgfile, "../data1/台湾猴")
    i = i+1
print(i)