import useful_functions
import script20240202_intensityevolutionin1area as intensity
import script20240224_intensitydropanalysis as intensitydrop
import script20240207_wormtracking as tracking
import time
import numpy as np
import os

test_pipeline = True
regenerate_tracking = False

tic = time.time()
if useful_functions.is_linux():
    if test_pipeline:
        path = "/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/subtest_for_tests/"
    else:
        path = '/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/'
else:
    if test_pipeline:
        path = 'E:/Backup/Patch_depletion_dissectoscope/subtest_for_tests/'
    else:
        path = 'E:/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/'

if regenerate_tracking:
    t, x, y, sil = tracking.generate_tracking(path, regenerate_assembled_images=False, track_assembled=True)
    print("Tracking over and saved, it took ", time.time() - tic, "seconds to run!")
else:
    os.chdir(path)
    t, x, y = np.load("list_tracked_frame_numbers.npy"), np.load("list_positions_x.npy"), np.load("list_positions_y.npy")
    print("Finished loading data tables.")

images_path = useful_functions.find_path_list(path)
assembled_images_path = useful_functions.find_path_list(path + "assembled_images/")

# intensity_evolution_1_area_where_worm_was(x, y, t, 34, 400, 500)
# intensity_as_a_function_of_worm_distance(path + "assembled_images/", [1000, 1500], [2500, 3000], x, y, t)
# useful_functions.interactive_worm_plot(assembled_images_path, t, x, y)
intensity.interactive_intensity_plot(path, None, None, analysis_size=1)
# intensitydrop.plot_intensity_worm_passages(path, t)
