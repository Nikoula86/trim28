import pandas as pd
import xml.etree.ElementTree as et 
import tqdm, os

#####################

pc = os.environ['COMPUTERNAME']
if pc=='PCBA-TRIVEDI03': # my Razer
    folder_raw = os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','immuno_NMG')
elif pc=='PCBA-TRIVEDI02': # workstation
    folder_raw = os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','immuno_NMG')

exp_folder = os.path.join('2021-02-05_NGM_immuno')

#####################

xtree = et.parse(os.path.join(folder_raw,exp_folder,'Images','Index.idx.xml'))
xroot = xtree.getroot()
# print(xroot.attrib['Images'])
images = xroot.findall('{http://www.perkinelmer.com/PEHH/HarmonyV5}Images')[0]
print(len(images))

df = pd.DataFrame({'filename':[], 
                    'Xpos':[], 'Ypos':[], 'Zpos':[], 
                    'row':[], 'col':[], 
                    'field':[], 'plane':[], 
                    'channel':[], 'chName':[], 
                    'expTime':[]})

for image in tqdm.tqdm(images.iter('{http://www.perkinelmer.com/PEHH/HarmonyV5}Image')):
    # print(image.tag, image.attrib)

    row = {}
    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}URL')
    row['filename'] = x.text

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionX')
    row['Xpos'] = float(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionY')
    row['Ypos'] = float(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PositionZ')
    row['Zpos'] = float(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}Row')
    row['row'] = int(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}Col')
    row['col'] = int(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}FieldID')
    row['field'] = int(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}PlaneID')
    row['plane'] = int(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelID')
    row['channel'] = int(x.text)

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}ChannelName')
    row['chName'] = x.text

    x = image.find('{http://www.perkinelmer.com/PEHH/HarmonyV5}ExposureTime')
    row['expTime'] = float(x.text)

    df = df.append(pd.Series(row), ignore_index=True)

print(df.head())
df.to_csv(os.path.join(folder_raw,exp_folder,'metadata.csv'))
