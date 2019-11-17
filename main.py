import sys
from dataset import *

sys.path.insert(0, './data')
import sitedict
from sitedict import *


def main():
    dataset = DataSet()
    # print(dataset.get_colnames())
    # print (SITEDICT)
    before = len(dataset.df.index)
    new_df = dataset.get_df_of_radius(1)
    after = len(new_df.index)
    print(f"Proportion remaining: {after/before}")


if __name__ == "__main__":
    main()
