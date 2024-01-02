
def segment_labely(ij,folder_in,segmented,processed,classifier,macro_path,N=3):
  print("INCOMPLETE IDEA - \n")
  print("  Was planning to use 'pyimagej' to import imagej,\n")
  print("and then use ij.run_plugin to operate inside python.\n")
  print("Now have found method to get macro to run so not finishing htis now.\n")
  # example code for operations we might need.
  #image = ij.io().open('sample-data/test_image.tif')
  #ij_image = ij.WindowManager.getCurrentImage()
  #ij.py.show(ij_image)
  # plugin name is not known, its probably a long dotted java class path, eg sci.java.kit.labkit .... or something.
  #ij.py.run_plugin("LabKitSegmenter")


def save_classifier_macro(folder_in,segmented,processed,classifier,macro_path,N=3):
  #Save the ij_macro to a macro script and later call this macro_command.

  # Path flop because windows slashes cause trouble.
  # this would break any escaped spaces, but I dont think there should be any.
  folder_in=folder_in.replace('\\','/')
  segmented=segmented.replace('\\','/')
  processed=processed.replace('\\','/')
  classifier=classifier.replace('\\','/')

  # When needing to test the macro in fiji, set the reusable macro to False.
  # That will hard-code the paths into the macro.
  #
  # Making the macro reusable is good to prevent issues with multiple scripts running this code.
  #
  ij_reusable_macro=True
  args=None
  if ij_reusable_macro:
    args = {
        "folder_in" : folder_in,
        "segmented" : segmented,
        "processed" : processed,
        "classifier" : classifier,
        "N" : N
    }
    # WARNING: The #@ lines MUST be left aligned no matter the python indent.
    #          The #@ lines define the macro variables
    #          I think these lines need to be at the beginning of the macro
    #          Leading blank lines appear okay.
    ij_var_section="""
#@ String folder_in
#@ String segmented
#@ String processed
#@ String classifier
#@ int N
"""
  else:
    ij_var_section="""
folder_in="{folder_in}";
segmented="{segmented}";
processed="{processed}";
classifier="{classifier}";
N={N}
""".format(folder_in=folder_in,segmented=segmented,processed=processed,classifier=classifier,N=str(N))

  ij_macro = ij_var_section + '''
    File.makeDirectory(segmented);
    File.makeDirectory(processed);
    for (i = 0; i < N; i++) {
        print(folder_in+"/Region_"+i+".tif");
        open(folder_in+"/Region_"+i+".tif");
        print("seg");
        run("Segment Image With Labkit", "segmenter_file="+classifier+" use_gpu=false");
        // run("Segment Image With Labkit", "segmenter_file="+s+" input=Region_"+i+".tif use_gpu=false");
        close("Region_"+i+".tif");
        print("Images open: "+nImages);
        // selectImage(1);
        saveAs("Tiff", segmented + "/segmentation_" + i + ".tif");
        print("bin");
        run("Make Binary", "calculate black");
        print("wat");
        run("Watershed","stack");
        print("con");
        run("Connected Components Labeling", "connectivity=6 type=[16 bits]");
        print("sav");
        saveAs("Tiff", processed + "/morpho_" + i + ".tif");
        close("*");
    }
    // this exit command doesn't always work. When using pyimagej it seems to have no effect.
    exit();
'''
  with open(macro_path,'w') as macro_handle:
    macro_handle.write(ij_macro)

  return ij_macro,args

