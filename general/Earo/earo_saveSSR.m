function earo_saveSSR(eobj, filename, norml, hpstruct, bpb)

    % saveSSR(eobj, filename, norml, [hpobj])
    %
    % based on miroToSSR by Benjamin Bernschutz
    % edited by Jonathan Sheaafer
    %
    % edited by Zamir Ben-Hur 21.3.18 - adding support of 2 channels headphone compensation
    % edited by Zamir Ben-Hur 25.7.18 - adding support of saving bpb file for playback in SSR (saving 718 channels)
    
    
    if nargin<5
        bpb = false;
    end
    if (strcmp(eobj.type,'BRIR')  || strcmp(eobj.type,'HRIR')) && eobj.nData == 360

        if ~eobj.shutUp && nargin>=4, fprintf('Applying headphone compensation, %s',hpstruct.hpName); end

        %irArray = zeros(size(eobj.data,1), 720);
        inc = 1;

        for i = 1:360
             if i==1, fprintf(' -- 000%%'); end
             if rem(i,floor(360/100))==0, fprintf('\b\b\b\b%03d%%',ceil((i/360)*100)); end
            
            IR = squeeze(eobj.data(i,:,:));

            if nargin>=4        % Headphone compensation
                if hpstruct.fs == eobj.fs
                    if size(hpstruct.minPhase,2)>1
                        irArray(:,inc) = miro_fastConv(IR(:,1),hpstruct.minPhase(:,1)); inc = inc+1;
                        irArray(:,inc) = miro_fastConv(IR(:,2),hpstruct.minPhase(:,2)); inc = inc+1; 
                    else
                        irArray(:,inc) = miro_fastConv(IR(:,1),hpstruct.minPhase); inc = inc+1;
                        irArray(:,inc) = miro_fastConv(IR(:,2),hpstruct.minPhase); inc = inc+1; 
                    end
                else
                    fprintf ('Headphone filter and earo object sample rate mismatch!\nDoing nothing.\n')
                end
            else
                irArray(:,inc) = IR(:,1);
                inc = inc+1;
                irArray(:,inc) =  IR(:,2);
                inc = inc+1;
            end

        end
        
        fprintf('\n');

        if norml, irArray = 0.99*irArray/max(max(abs(irArray))); end
        if bpb, irArray = irArray(:,1:718); end
        wavwrite(irArray, eobj.fs, 16, filename)
        disp(['SSR file sucessfully generated: ', filename])
    else
        if ~eobj.shutUp
            fprintf('Failed: SSR export requires circular HRIR or BRIR objects.\n');
        end
        return
    end

end

function ab = miro_fastConv(a,b)

% Internal use only

NFFT = size(a,1)+size(b,1)-1;
A    = fft(a,NFFT);
B    = fft(b,NFFT);
AB   = A.*B;
ab   = ifft(AB);

end