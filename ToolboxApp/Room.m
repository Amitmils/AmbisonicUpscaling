classdef Room
    % 
    
    properties
        RoomSize = [5 5 5];
        ArrayPosition = [2 2 2];
        Reflections = 0.9
    end
    
    methods
        function obj = Room(RoomSize,ArrayPosition,Reflections)
            
            if nargin >0
                obj.RoomSize = RoomSize;
                obj.ArrayPosition = ArrayPosition;
                obj.Reflections = Reflections;
            end
        end
        function obj = Load(obj,file)
            % Load room from file
            try
                fid = fopen(file);
                raw = fread(fid,inf);
                str = char(raw'); 
                fclose(fid); 
                decoded = jsondecode(str);
                
                obj.RoomSize = decoded.RoomSize';
                obj.ArrayPosition = decoded.ArrayPosition';
                
                obj.Reflections = decoded.Reflections;
                
            catch
                warndlg("Bad json file")
            end
        end
        function Save(obj,path)
            %Save room to file
            fid =fopen(path,'w') ;
            encoded = jsonencode(obj);
            fprintf(fid, encoded);
        end
    end
end

