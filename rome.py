import os
from multiprocessing import Pool

class center:
    def __init__(self,host_list,user_list,n_proc=6):
        self.host_list=host_list
        self.user_list=user_list
        self.uh_list = list([u,h] for u,h in zip(user_list,host_list))
        self.h2u = dict((h,u) for u, h in zip(user_list, host_list))
        self.p=Pool(n_proc)

    def exe(self,cmd):
        self.p.map(lambda x:os.system("ssh -l %s %s '%s'"%(x[0],x[1],cmd)),
                   self.uh_list)

    def spread(self,path):
        self.p.map(lambda x:os.system("scp -r %s %s@%s:%s"%(path,x[0],x[1],path)),
                   self.uh_list)

    def hexe(self,host,cmd):
        os.system("ssh -l %s %s '%s'" % (self.h2u[host],host, cmd))

    def hspread(self,host,path):
        os.system("scp -r %s %s@%s:%s"%(path,self.h2u[host],host,path))

class cluster(center):
    def __init__(self,conf):
        """
        :param conf: right the conf file in format of: user,host for each line
        """
        self.conf=conf
        self.user_list,self.host_list=zip([i.split(",")] for i in list(open(self.conf).read()))
        self.user_list=list(self.user_list)
        self.host_list=list(self.host_list)
        super(cluster,self).__init__(self.host_list,self.user_list)
