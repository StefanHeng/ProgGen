"""
For the CoNLL2003 dataset
"""


if __name__ == '__main__':
    from stefutil import pl, sic

    def get_entity_counts():
        """
        Counts number of annotated entities in each dataset split
        """
        counts = dict(
            train=dict(LOC=7_140, MISC=3_438, ORG=6_321, PER=6_600),
            dev=dict(LOC=1_837, MISC=922, ORG=1_341, PER=1_842),
            test=dict(LOC=1_668, MISC=702, ORG=1_661, PER=1_617)
        )
        lbs = ['LOC', 'MISC', 'ORG', 'PER']
        split2counts = {split: sum([counts[split][lb] for lb in lbs]) for split in counts}
        sic(split2counts)
    # get_entity_counts()

    counts_test = dict(person=1_617, location=1_668, organization=1_661, misc=702)
    print(pl.i(counts_test))
    # normalize and round to 2 decimal places percents
    counts_test_norm = {k: round(v / sum(counts_test.values()) * 100, 2) for k, v in counts_test.items()}
    print(pl.i(counts_test_norm))
