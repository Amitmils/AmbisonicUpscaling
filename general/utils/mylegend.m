function legend_hanle = mylegend(varargin)
% This function improves the built-in legend function of matlab, by
% allowing interactive interface. Clicking on the values in the legend,
% will mark the relevant plot in the axes.
%
% Possible gestures:
%   one click - mark/unmark
%   double click - mark only this one / reset all
%
% Possible marking modes:
%   '-visible' - changing visibility
%   '-bold' - changing line-width
%
%
% ---INPUT---
%   all argument are passed to the built in legend function. controling of
%   the marking mode, is done by adding a string '-visible' / '-bold'
% ---OUTPUT---
%   legend handle

% Author: Ran Weissman, ACLab BGU, 2019



state_type = '-visible'; %default value
i_delete = [];

%default values for '-bold' mode
width_on = 3;
% width_off = 0.5;

%look if user's input containing legend's type
for i = 1:length(varargin)
    if ischar(varargin{i}) && ( strcmpi(varargin{i},'-visible') || strcmpi(varargin{i},'-bold') )
        state_type = varargin{i};
        i_delete = i;
    end
end

varargin(i_delete) = [];


%create legend
legend_hanle = legend(varargin{:});
legend_hanle.ItemHitFcn = @hitcallback_all;



%% callback function for interactive legend

%main callback for changing states
function hitcallback_all(~,evnt)
obj = evnt.Peer;

switch evnt.SelectionType
    case 'normal' %single left click
        if check_state(obj)
            change_state(obj,0)
        else
            change_state(obj,1)
        end
    case 'open' %double left click
        lines_list = evnt.Peer.Parent.Children;
        if check_state(obj)
            change_state(lines_list,0);
            change_state(obj,1);
        else
            counter = 0;
            for j = 1:length(lines_list)
                if check_state(lines_list(j))
                    counter = counter + 1;
                end
            end
            if counter == 0
                change_state(lines_list,1);
            else
                change_state(lines_list,0);
                change_state(obj,1);
            end
        end
end

end


%function for checking state
function state = check_state(obj)
switch state_type
    case '-visible'
        switch obj.Visible
            case 'on'
                state = 1;
            case 'off'
                state = 0;
        end
    case '-bold'
        switch obj.LineWidth
            case width_on
                state = 1;
            otherwise
                state = 0;
        end
end
end

%function for changing state
function change_state(obj,state)
switch state_type
    case '-visible'
        switch state
            case 1
                set(obj,'Visible','on');
            case 0
                set(obj,'Visible','off');
            otherwise
                error('unknown state');
        end
    case '-bold'
        switch state
            case 1
%                 set(obj,'LineWidth',width_on);
                obj.LineWidth = obj.LineWidth*width_on;
            case 0
%                 set(obj,'LineWidth',width_off);
                obj.LineWidth = obj.LineWidth/width_on;
            otherwise
                error('unknown state');
        end
end
end



end


