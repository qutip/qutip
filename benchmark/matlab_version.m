function []=matlab_version()
v=version;
comp=computer;

mat_ver={v,comp};

fid = fopen('matlab_version.csv','w');

for row = 1:size(mat_ver,1)
    fprintf(fid, repmat('%s,',1,size(mat_ver,2)-1), mat_ver{row,1:end-1});
    fprintf(fid, '%s\n', mat_ver{row,end});
end
fclose(fid);