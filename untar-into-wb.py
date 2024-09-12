import os

charge = [2,3,4,5,6,7,8]
root = 'Wb-qm0'

for q in charge :
    folder =  "WATER-butanol-qm0"+str(q)
    os.system("mkdir "+folder)
    fn1 = root+str(q)+'-1.tar.gz'
    fn2 = root+str(q)+'-1.tar.gz'
    os.system("tar -xvf "+fn1)
    os.system("mv Flow/*.dat "+folder)
    os.system("tar -xvf "+fn2)
    os.system("mv Flow/*.dat "+folder)
