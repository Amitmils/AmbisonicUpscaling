classdef signal_holder
    properties
        name
        signal
        fs
    end
    methods
        function obj = signal_holder(name,signal,fs)
            obj.name = name;
            obj.signal = signal;
            obj.fs = fs;
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