from xlwt import *
import re
# 需要xlwt库的支持
# import xlwt
file = Workbook(encoding='utf-8')
# 指定file以utf-8的格式打开



for i in range(11):
    fp =open("/media/liqiang/windata/project/classification/plugin/experiment"+str(i+1)+".log")
    netlist = ['mobilenet', 'resnet', 'shufflenet', 'squeezenet', 'alexnet', 'densenet', 'googlenet', 'MNASNet',
               'VGG']
    taindex = 0
    idex = 1
    table = file.add_sheet("experiment"+str(i+1))
    for line in fp.readlines(): # 遍历每一行
        filename = line[:14]    # 每行取前14个字母，作为下面新建文件的名称

        if line[1:3] in netlist[0]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[1]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[2]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[3]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[4]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[5]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[6]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[7]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4
        elif line[1:3] in netlist[8]:
            content = line[:]     # 每行取第15个字符后的所有字符，作为新建文件的内容
            # print(content[:-2])
            if idex==1:
                table.write(0, taindex+0, content[:-2])
                table.write(0, taindex+1, 'loss')
                table.write(0, taindex+2, 'acc')
                taindex +=4
            elif idex==301:
                table.write(0, taindex + 0, content[:-2])
                table.write(0, taindex + 1, 'loss')
                table.write(0, taindex + 2, 'acc')
                taindex += 4

        # with open("e:\\"+filename+".txt","w") as fp2:
        #     fp2.write(content+"\n")

        if line[:5]=='Batch':
            num = re.findall(r'\d+', line)
            if int(num[0])<=300:
                if int(idex) <= 300:
                    # print(line)
                    table.write(int(num[0]), taindex+1-4, line[line.find(':')+1:line.find(':')+9])
                    table.write(int(num[0]), taindex+2-4, line[line.rfind(':') + 1:line.rfind(':') + 6])
                    idex +=1
                elif int(idex) > 300:
                    idex=1
                    # print(line)
                    table.write(int(num[0]), taindex+1-4, line[line.find(':') + 1:line.find(':') + 9])
                    table.write(int(num[0]), taindex+2-4, line[line.rfind(':') + 1:line.rfind(':') + 6])
                    idex += 1
                print('idex:',idex)
    fp.close()
    file.save('data.xlsx')
