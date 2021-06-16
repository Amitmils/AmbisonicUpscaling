classdef signal_holder
    properties
        name
        signal
        fs
        position
    end
    methods
        function obj = signal_holder(name,signal,fs)
            obj.name = name;
            obj.signal = signal;
            obj.fs = fs;
            obj.position = [10 3 1.7];
        end
        function obj = set_position(obj, position)
             obj.position = position;
        end
        function obj = play(obj,device)
             mindata = min(min(obj.signal));
             maxdata = max(max(obj.signal));
             normalised = ((obj.signal-mindata)*(1-(-1))/((maxdata)-mindata))+(-1);
             player = audioplayer(normalised,obj.fs,16,device);
             playblocking(player);
        end
        function obj = save(obj)
            [savefile,savepath] = uiputfile(strcat(obj.name,'.wav'));
            if savefile
                audiowrite(strcat(savepath,savefile),obj.signal,obj.fs);
            end
        end
        function obj = rename(obj,name)
            obj.name = name;
        end
    end
    
end