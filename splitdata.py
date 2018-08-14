import os,shutil,random

def movefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s" % (srcfile,dstfile))

if __name__ == '__main__':
    read_dir = "all_result/result"
    dst_dir = "all_result/test"
    for filename in os.listdir(read_dir):
        if random.uniform(0 ,1) > 0.85:
            movefile(os.path.join(read_dir, filename), os.path.join(dst_dir, filename))