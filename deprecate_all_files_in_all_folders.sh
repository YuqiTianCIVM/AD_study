parent_folder="/s/yt133/To_delete_Statistics/data_5xFAD_Abeta_counting"
for subdir in "$parent_folder"/*/; do
    if [ -d "$subdir" ]; then
        echo "Processing $subdir"
        cd "$subdir" && mkdir -p 2023_counted && find . -maxdepth 1 -not -name '2023_counted' -not -name '.' -exec mv {} 2023_counted/ \;
    fi
done