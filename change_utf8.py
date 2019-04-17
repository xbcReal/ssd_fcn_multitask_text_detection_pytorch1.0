import chardet
import codecs
import os
def convert_file_to_utf8(filename):
    # !!! does not backup the origin file
    content = codecs.open(filename, 'r').read()
    print(type(content))
    source_encoding = chardet.detect(content.encode('utf8'))['encoding']
    if source_encoding == None:
        print("??",filename)
        return
    print("  ",source_encoding, filename)
    #if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
    #content = content.decode(source_encoding, 'ignore') #.encode(source_encoding)
    codecs.open(filename, 'w', encoding='UTF-8-SIG').write(content)
files = os.listdir('/home/binchengxiong/ocr_data/ICDAR2015/ch4_training_localization_transcription_gt_for_eval')
for i in files:
    print(i)
    convert_file_to_utf8(os.path.join('/home/binchengxiong/ocr_data/ICDAR2015/ch4_training_localization_transcription_gt_for_eval/',i))