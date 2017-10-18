from datetime import datetime
class lprint(object):
    def __init__(self,task="task"):
        """
        Standard log printing mechanism
        l=lprint("mytask")
        When instantiate, input task name
        Then l.print("do_this","the log content of that")
        """
        self.time0=datetime.now()
        self.timelast=datetime.now()
        self.task=task
        self.p("task:%s>>start"%(self.task))

    def marktime(self):
        """
        Mark the current time
        """
        return datetime.now().strftime("%Y-%m-%d_%H:%I:%S")

    def fromlast(self):
        """
        The timespan from last mark
        """
        rt=(datetime.now()-self.timelast).seconds
        self.timelast=datetime.now()
        return rt

    def fromstart(self):
        """
        The timespan from starting
        """
        return (datetime.now()-self.time0).seconds

    def p(self,title="log",content=""):
        """
        print the log line
        """
        print("[%s]<%s|%ss,%ss>\t%s"%(title,
                                  self.marktime(),
                                    self.fromlast(),
                                    self.fromstart(),
                                  content,
                                 ))