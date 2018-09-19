function f_struct = write_txt(f_struct, root_name, folder_name, video_name, save_bbox, save_confidence, save_time, f_flag)

switch f_flag
    
    case 1 % open
        
        save_folder_name = [root_name, folder_name, '/longterm/', video_name, '/'];

        if ~isdir(save_folder_name)
            mkdir(save_folder_name);
        end

        name_bbox = [video_name, '_001.txt'];
        name_conf = [video_name, '_001_confidence.value'];
        name_time = [video_name, '_time.txt'];

        f_struct.file_bbox = fopen([save_folder_name, name_bbox], 'wt');
        f_struct.file_conf = fopen([save_folder_name, name_conf], 'w');
        f_struct.file_time = fopen([save_folder_name, name_time], 'w');
        
    case 2 % write
        
        if sum(isnan(save_bbox)) == 0
            str_bbox = [];
            if numel(save_bbox) == 4
                for i = 1 : numel(save_bbox)
                    tmp = num2str(save_bbox(i), '%4.4f');
                    str_bbox = cat(2, str_bbox, tmp);
                    if i ~= numel(save_bbox), str_bbox = cat(2, str_bbox, ','); end
                end
                fprintf(f_struct.file_bbox, '%s\n', str_bbox);
            else
                fprintf(f_struct.file_bbox, '1\n');
            end
        else
%             fprintf(f_struct.file_bbox, '\nnan,nan,nan,nan');
            fprintf(f_struct.file_bbox, '-1.0000,-1.0000,-1.0000,-1.0000\n');
        end

        if isnan(save_confidence) == 0
            str_confidence = num2str(save_confidence, '%4.6f');
            fprintf(f_struct.file_conf, '%s\n', str_confidence);
        else
            fprintf(f_struct.file_conf, '0.000000\n');
        end
        
        str_time = num2str(save_time, '%4.5f');
        fprintf(f_struct.file_time, '%s\n', str_time);
        
    case 3 % close
        fclose(f_struct.file_bbox);
        fclose(f_struct.file_time);
        fclose(f_struct.file_conf);
        f_struct = [];
end
    

end