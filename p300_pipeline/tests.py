import time
from user_profile import UserProfile, ClassifierType
from user_profile_util import SpatialFilters


def test_truncate():
    features = [1, 2, 3, 4, 5, 6]
    labels = [True, False, False, False, True, False]
    new_features, new_labels = UserProfile.truncate(features, labels)
    new_features_expected = [1, 2, 3, 5]
    new_labels_expected = [True, False, False, True]
    print '{} new_features\n{} new_features_expected'.format(
        new_features, new_features_expected
    )
    print '{} new_labels\n{} new_labels_expected'.format(
        new_labels, new_labels_expected
    )


def test_classifiers():
    profile_filenames = ['player1_2016_10_09_16_32',
                         'player1_2016_11_10_12_40',
                         'player1_2016_11_10_15_18']

    profile_filenames = [
        'felipevieirafrujeri_2016_11_12_15_04',
        'felipevieirafrujeri_2016_11_12_15_13',
        'felipevieirafrujeri_2016_11_12_16_36',
        'felipevieirafrujeri_2016_11_12_16_40',
        'felipevieirafrujeri_2016_11_12_16_43',
        'felipevieirafrujeri_2016_11_12_16_54',
        'felipevieirafrujeri_2016_11_12_18_26'
    ]

    # TODO: print settings.xml too!
    # TODO: better select optimal model: num_filters, classifier_type, epoch_duration
    output = open('../calibration_stats_unequal.csv', 'w')
    # helmet inverted if profile_filename = '{}_inv'
    output.write('profile_filename,helmet_positioning,num_epochs,classifier,'
                 'computing_time,' +
                 'epoch_duration,epoch_size,num_filters,test_size,'
                 'accuracy,targets,non_targets,' +
                 'TP,FP,TN,FN,(TP/(TP+FP),TN/(TN+FN),TP/(TP+FN),TN/(TN+FP)\n')
    for profile_filename in profile_filenames:
        helmet_inverted = '_inv' in profile_filename
        for classifier_type in ClassifierType:
            for epoch_duration in [0.600, 0.700, 0.875]:
                for test_size in [0.2, 0.3]:
                    for num_filters in range(1, SpatialFilters.MAX_NUM_FILTERS + 1):
                        computing_time = time.time()
                        user = UserProfile()
                        user.compute_profile(profile_filename=profile_filename,
                                             classifier_type=classifier_type,
                                             epoch_duration=epoch_duration,
                                             test_size=test_size,
                                             num_filters=num_filters,
                                             equal_labels_ratio=False)
                        computing_time = time.time() - computing_time
                        output.write(
                            '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                                profile_filename,
                                'inv' if helmet_inverted else 'normal',
                                user.num_epochs,
                                classifier_type,
                                computing_time,
                                user.epoch_duration,
                                user.epoch_size,
                                num_filters,
                                test_size,
                                user.accuracy,
                                user.true_positives + user.false_negatives,
                                user.true_negatives + user.false_positives,
                                user.true_positives,
                                user.false_positives,
                                user.true_negatives,
                                user.false_negatives,
                                ratio(user.true_positives, user.false_positives),
                                ratio(user.true_negatives, user.false_negatives),
                                ratio(user.true_positives, user.false_negatives),
                                ratio(user.true_negatives, user.false_positives)
                            )
                        )
            print 'Done with {} for {}.'.format(classifier_type,
                                               profile_filename)

    output.close()


def ratio(x, y):
    if x + y == 0:
        return 'nan'
    else:
        return 100 * x / (x + y)


if __name__ == '__main__':
    test_classifiers()
    # test_truncate()
