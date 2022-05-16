import random


class MixIter:
    def __init__(self, iter_list, mix_step=1):
        if not iter_list:
            raise Exception("Empty iter list")
        self.acc_length = [len(iter_list[0])]
        for i in iter_list[1:]:
            self.acc_length.append(len(i)+self.acc_length[-1])
        self.total_num = self.acc_length[-1]
        self.iter_list = iter_list
        self.mix_step = mix_step

        # shuffle nums
        self.nums = []
        last = 1
        for n in self.acc_length:
            segs = self.get_segs(list(range(last, 1+n)))
            self.nums+=segs
            last = n+1
        random.shuffle(self.nums)
        self.nums = [i for sub_li in self.nums for i in sub_li]

    def get_segs(self, nums):
        li = []
        n = len(nums)
        i, end = 0, n-self.mix_step
        while i < end:
            li.append(nums[i:i+self.mix_step])
            i += self.mix_step
        else:
            li.append(nums[i:])
        return li


    def findUpper(self, num):
        for i, acc in enumerate(self.acc_length):
            if num <= acc:
                return i

    def __iter__(self):
        self.cur_iter_list = [iter(i) for i in self.iter_list]
        self.cur = 0
        return self

    def __next__(self):
        if self.cur == self.total_num:
            raise StopIteration
        else:
            num = self.nums[self.cur]
            self.cur += 1
            idx = self.findUpper(num)
            return next(self.cur_iter_list[idx]), idx  # todo

    def __getattr__(self, attr):
        return getattr(self.iter_list[0], attr)

    def __len__(self):
        return self.total_num


if __name__ == "__main__":
    l1 = (1, 2, 3, 4, 5)
    l2 = 'abcdef'
    l3 = ['@', '#', '$']
    mi = MixIter([l1, l2, l3], 4)
    for i in mi:
        pass
    for i in mi:
        print('iter', i)
    print('length', len(mi))
