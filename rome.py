import os

class center:
    def __init__(self,host_list,user_list):
        self.host_list=host_list
        self.user_list=user_list
        self.uh_list = list([u,h] for u,h in zip(user_list,host_list))
        self.h2u = dict((h,u) for u, h in zip(user_list, host_list))

    def exe(self,cmd):
        result=list(map(lambda x:os.system("ssh -l %s %s '%s'"%(x[0],x[1],cmd)),
                   self.uh_list))
        os.system(cmd)
        list(print("[%s:result]"%(h),"\n",) for h,r in zip(self.host_list,result))

    def spread(self,path):
        result=list(map(lambda x:os.system("scp -r %s %s@%s:%s"%(path,x[0],x[1],path)),
                   self.uh_list))
        list(print("[%s:result]" % (h), "\n", ) for h, r in zip(self.host_list, result))

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
