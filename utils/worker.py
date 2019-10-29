import logging
import threading

logger = logging.getLogger('[WORKER]')
logger.setLevel(logging.INFO)

__author__ = 'Andres'

'''

This class simplifies thread usage. Examples:

1 -	Worker.call(aFunction).withArgs(arg1, arg2..argN).start() / Runs a normal thread starting at aFunction

2 -	Worker.call(aFunction).withArgs(arg1, arg2..argN).asDaemon.start() / Same as before, but uses a daemon thread

3 - Worker.call(aFunction).withArgs(arg1, arg2..argN).every(T).asDaemon.start() / Runs a thread every T seconds

4 - Worker.call(aFunction).withArgs(arg1, arg2..argN).after(T).asDaemon.start() / Runs a thread after T seconds

NOTE: The 'call' method should be called first ALWAYS!!

CronicWorker - Calling the 'every(seconds)' function returns a CronicWorker with the original Worker attributes.
DeferredWorker - Calling the 'after(seconds)'  function returns a DeferredWorker with the original Worker attributes.

NOTE: Calling 'start()' more than once on a DeferredWorker will try to 'cancel()' the first thread before launching
a new one

'''


class Worker(object):
    def __init__(self):
        self._thread = None
        self._isDaemon = False
        self._function = None
        self._callback = lambda: None
        self._arguments = ()

    @staticmethod
    def call(function):
        worker = Worker()
        worker._function = function
        return worker

    def withArgs(self, *args):
        self._arguments = args
        return self

    @property
    def asDaemon(self):
        self._isDaemon = True
        return self

    def start(self):
        self._thread = threading.Thread(target=self._startPoint)
        self._thread.daemon = self._isDaemon
        self._thread.start()

        return self

    def isWorking(self):
        return self._thread.isAlive() if self._thread else False

    def join(self, timeout=None):
        if self.isWorking():
            self._thread.join(timeout)

        return self

    # def every(self, seconds):
    #     from utils.worker.cronicWorker import CronicWorker
    #     cronicWorker = CronicWorker.fromWorker(self)
    #     cronicWorker._repeatInterval = seconds
    #     return cronicWorker
    #
    # def after(self, seconds):
    #     from utils.worker.deferredWorker import DeferredWorker
    #     deferredWorker = DeferredWorker.fromWorker(self)
    #     deferredWorker._waitTime = seconds
    #     return deferredWorker

    def _startPoint(self):
        logger.debug("Worker <%s> is about to call: %s%s", self._thread.ident, self._function.__name__,
                     str(self._arguments))

        self._function(*self._arguments)

        logger.debug("Worker <%s> called: %s%s", self._thread.ident, self._function.__name__,
                     str(self._arguments))

    def _reset(self):
        self._thread = None
        self._isDaemon = False
        self._function = None
        self._callback = lambda: None
        self._arguments = ()
