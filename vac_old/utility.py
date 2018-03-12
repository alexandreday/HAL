import os

class FOUT:
    def __init__(self, fname='tmp.txt'):
        self.fname = 'tmp.txt'
        if os.path.isfile(fname):
            os.system("rm %s"%self.fname) # removes file if it exist

    def write(self, mystring): # this will append
        fout = open(self.fname,'a')
        fout.write(mystring+'\n')
        fout.close()