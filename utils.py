def progressbar(i, upper_range):
    percentComplete = int(i / (upper_range) * 100)

    # Some functions in this project take some time to run due to loops.
    # This gives visual indication of progress
    progress_string = f'\r{("#" * percentComplete)}{("_" * ((100) - percentComplete))} {percentComplete} / {100} [ Printing Frame: {i} ]'
    if i == upper_range:
        print(progress_string)
    else:
        print(progress_string, end='')
