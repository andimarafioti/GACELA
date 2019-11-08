import time


class ConsoleSummarizer(object):
	def __init__(self, log_interval, batch_size, datasets_length):
		super(ConsoleSummarizer, self).__init__()

		self._log_interval = log_interval
		self._batch_size = batch_size
		self._datasets_length = datasets_length

		self._start_time = time.time()
		self._prev_iter_time = self._start_time

	def printSummary(self, batch_idx, epoch):
		current_time = time.time()

		print(" * Epoch: [{:2d}] [{:4d}/{:4d} ({:.0f}%)] "
			  "Counter:{:2d}\t"
			  "({:4.1f} min\t"
			  "{:4.3f} examples/sec\t"
			  "{:4.2f} sec/batch)\n".format(
			int(epoch),
			int(batch_idx),
			int(self._datasets_length), 100. * batch_idx / self._datasets_length, int(batch_idx),
														(current_time - self._start_time) / 60,
														self._log_interval * self._batch_size /
														(current_time - self._prev_iter_time),
														(current_time - self._prev_iter_time) / self._log_interval))
		self._prev_iter_time = current_time
