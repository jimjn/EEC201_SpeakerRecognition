classdef final_gui_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                       matlab.ui.Figure
        EEC201SpeakerRecognitionLabel  matlab.ui.control.Label
        UIAxesTrain                    matlab.ui.control.UIAxes
        UIAxesTest                     matlab.ui.control.UIAxes
        RecordButtonTrain              matlab.ui.control.Button
        RecordButtonTest               matlab.ui.control.Button
        UIAxesClassify                 matlab.ui.control.UIAxes
        ClassifyButton                 matlab.ui.control.Button
        RecordingLampLabel             matlab.ui.control.Label
        RecordingLampTrain             matlab.ui.control.Lamp
        RecordingLamp_2Label           matlab.ui.control.Label
        RecordingLampTest              matlab.ui.control.Lamp
        PlayButtonTrain                matlab.ui.control.Button
        PlayButtonTest                 matlab.ui.control.Button
        recObj
        recordedInputTrain
        recordedInputTest
        %codebooks
        c1
        c2
        c3
        c4
        c5
        c6
        c7
        c8
        c9
        c10
        c11
        cTrain
    end

    methods (Access = private)

        % Button pushed function: RecordButtonTrain
        function RecordButtonTrainPushed(app, event)
            app.RecordingLampTrain.Color = 'green';
            app.recObj = audiorecorder(12500,16,1);
            disp('Start speaking.')
            recordblocking(app.recObj, 2);
            disp('End of Recording.');
            app.RecordingLampTrain.Color = 'red';
            app.recordedInputTrain = getaudiodata(app.recObj);
            time = [0:1/12500:length(app.recordedInputTrain)*(1/12500)-(1/12500)];
            plot(time,app.recordedInputTrain,'Parent', app.UIAxesTrain);
        end

        % Button pushed function: PlayButtonTrain
        function PlayButtonTrainPushed(app, event)
            soundsc(app.recordedInputTrain,12500);
        end

        % Button pushed function: RecordButtonTest
        function RecordButtonTestPushed(app, event)
            app.RecordingLampTest.Color = 'green';
            app.recObj = audiorecorder(12500,16,1);
            disp('Start speaking.');
            recordblocking(app.recObj,2);
            disp('End of Recording.');
            app.RecordingLampTest.Color = 'red';
            app.recordedInputTest = getaudiodata(app.recObj);
            time = [0:1/12500:length(app.recordedInputTest)*(1/12500)-(1/12500)];
            plot(time,app.recordedInputTest,'Parent', app.UIAxesTest);
        end

        % Button pushed function: PlayButtonTest
        function PlayButtonTestPushed(app, event)
            soundsc(app.recordedInputTest,12500);
        end

        % Button pushed function: ClassifyButton
        function ClassifyButtonPushed(app, event)
            nfilt = 26;
            norm = 'y';
            lift = 'l';
            [cepsCoeff1] = mfcc(app.recordedInputTrain,12500,nfilt,norm,lift);
            [app.cTrain, p1, DistHist1]=vqsplit(cepsCoeff1',32);
            [cepsCoeffTest] = mfcc(app.recordedInputTest,12500,nfilt,norm,lift);
            similarityValues = similarityToAllCodebooks(cepsCoeffTest,length(cepsCoeffTest(:,1)'),26,32,app.cTrain,app.c2,app.c3,app.c4,app.c5,app.c6,app.c7,app.c8,app.c9,app.c10,app.c11);
            scatter([1:11],similarityValues,'filled','Parent',app.UIAxesClassify);
        end
    end

    % App initialization and construction
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure
            app.UIFigure = uifigure;
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'UI Figure';

            % Create EEC201SpeakerRecognitionLabel
            app.EEC201SpeakerRecognitionLabel = uilabel(app.UIFigure);
            app.EEC201SpeakerRecognitionLabel.FontSize = 20;
            app.EEC201SpeakerRecognitionLabel.Position = [244 416 279 24];
            app.EEC201SpeakerRecognitionLabel.Text = 'EEC 201 Speaker Recognition';

            % Create UIAxesTrain
            app.UIAxesTrain = uiaxes(app.UIFigure);
            title(app.UIAxesTrain, 'Training Input')
            xlabel(app.UIAxesTrain, 'X')
            ylabel(app.UIAxesTrain, 'Y')
            app.UIAxesTrain.PlotBoxAspectRatio = [1 0.154195011337868 0.154195011337868];
            app.UIAxesTrain.Position = [28 296 342 101];

            % Create UIAxesTest
            app.UIAxesTest = uiaxes(app.UIFigure);
            title(app.UIAxesTest, 'Test Input')
            xlabel(app.UIAxesTest, 'X')
            ylabel(app.UIAxesTest, 'Y')
            app.UIAxesTest.PlotBoxAspectRatio = [1 0.135964912280702 0.135964912280702];
            app.UIAxesTest.Position = [18 179 352 97];

            % Create RecordButtonTrain
            app.RecordButtonTrain = uibutton(app.UIFigure, 'push');
            app.RecordButtonTrain.ButtonPushedFcn = createCallbackFcn(app, @RecordButtonTrainPushed, true);
            app.RecordButtonTrain.Position = [385 335 60 38];
            app.RecordButtonTrain.Text = 'Record';

            % Create RecordButtonTest
            app.RecordButtonTest = uibutton(app.UIFigure, 'push');
            app.RecordButtonTest.ButtonPushedFcn = createCallbackFcn(app, @RecordButtonTestPushed, true);
            app.RecordButtonTest.Position = [383 216 62 42];
            app.RecordButtonTest.Text = 'Record';

            % Create UIAxesClassify
            app.UIAxesClassify = uiaxes(app.UIFigure);
            title(app.UIAxesClassify, 'Classification Results')
            xlabel(app.UIAxesClassify, 'Speaker # [Train speaker 2-11]')
            ylabel(app.UIAxesClassify, 'Sum Distance From Codebook')
            app.UIAxesClassify.PlotBoxAspectRatio = [1 0.506410256410256 0.506410256410256];
            app.UIAxesClassify.Position = [358 19 256 161];
            %app.UIAxes.label

            % Create ClassifyButton
            app.ClassifyButton = uibutton(app.UIFigure, 'push');
            app.ClassifyButton.ButtonPushedFcn = createCallbackFcn(app, @ClassifyButtonPushed, true);
            app.ClassifyButton.Position = [227 88 100 22];
            app.ClassifyButton.Text = 'Classify';

            % Create RecordingLampLabel
            app.RecordingLampLabel = uilabel(app.UIFigure);
            app.RecordingLampLabel.HorizontalAlignment = 'right';
            app.RecordingLampLabel.Position = [471 335 60 22];
            app.RecordingLampLabel.Text = 'Recording';

            % Create RecordingLampTrain
            app.RecordingLampTrain = uilamp(app.UIFigure);
            app.RecordingLampTrain.Position = [546 335 20 20];

            % Create RecordingLamp_2Label
            app.RecordingLamp_2Label = uilabel(app.UIFigure);
            app.RecordingLamp_2Label.HorizontalAlignment = 'right';
            app.RecordingLamp_2Label.Position = [471 216 60 22];
            app.RecordingLamp_2Label.Text = 'Recording';

            % Create RecordingLampTest
            app.RecordingLampTest = uilamp(app.UIFigure);
            app.RecordingLampTest.Position = [546 216 20 20];

            % Create PlayButtonTrain
            app.PlayButtonTrain = uibutton(app.UIFigure, 'push');
            app.PlayButtonTrain.ButtonPushedFcn = createCallbackFcn(app, @PlayButtonTrainPushed, true);
            app.PlayButtonTrain.Position = [385 305 100 22];
            app.PlayButtonTrain.Text = 'Play';

            % Create PlayButtonTest
            app.PlayButtonTest = uibutton(app.UIFigure, 'push');
            app.PlayButtonTest.ButtonPushedFcn = createCallbackFcn(app, @PlayButtonTestPushed, true);
            app.PlayButtonTest.Position = [385 187 100 22];
            app.PlayButtonTest.Text = 'Play';
            
            %set initial state of lamps
            app.RecordingLampTest.Color = 'red';
            app.RecordingLampTrain.Color = 'red';
            
            %Input 11 sample speakers, generate MFCC and Codebooks
            [s1,Fs(1)] = audioread('./Training_Data/s1.wav');
            [s2,Fs(2)] = audioread('./Training_Data/s2.wav');
            [s3,Fs(3)] = audioread('./Training_Data/s3.wav');
            [s4,Fs(4)] = audioread('./Training_Data/s4.wav');
            [s5,Fs(5)] = audioread('./Training_Data/s5.wav');
            [s6,Fs(6)] = audioread('./Training_Data/s6.wav');
            [s7,Fs(7)] = audioread('./Training_Data/s7.wav');
            [s8,Fs(8)] = audioread('./Training_Data/s8.wav');
            [s9,Fs(9)] = audioread('./Training_Data/s9.wav');
            [s10,Fs(10)]= audioread('./Training_Data/s10.wav');
            [s11,Fs(11)] = audioread('./Training_Data/s11.wav');
            
            nfilt = 26;
            norm = 'y';
            lift = 'l';
            [cepsCoeff1] = mfcc(s1,Fs(1),nfilt,norm,lift);
            [cepsCoeff2] = mfcc(s2,Fs(2),nfilt,norm,lift);
            [cepsCoeff3] = mfcc(s3,Fs(3),nfilt,norm,lift);
            [cepsCoeff4] = mfcc(s4,Fs(4),nfilt,norm,lift);
            [cepsCoeff5] = mfcc(s5,Fs(5),nfilt,norm,lift);
            [cepsCoeff6] = mfcc(s6,Fs(6),nfilt,norm,lift);
            [cepsCoeff7] = mfcc(s7,Fs(7),nfilt,norm,lift);
            [cepsCoeff8] = mfcc(s8,Fs(8),nfilt,norm,lift);
            [cepsCoeff9] = mfcc(s9,Fs(9),nfilt,norm,lift);
            [cepsCoeff10] = mfcc(s10,Fs(10),nfilt,norm,lift);
            [cepsCoeff11] = mfcc(s11,Fs(11),nfilt,norm,lift);
            
            [app.c1, p1, DistHist1]=vqsplit(cepsCoeff1',32);
            [app.c2, p2, DistHist2]=vqsplit(cepsCoeff2',32);
            [app.c3, p3, DistHist3]=vqsplit(cepsCoeff3',32);
            [app.c4, p4, DistHist4]=vqsplit(cepsCoeff4',32);
            [app.c5, p5, DistHist5]=vqsplit(cepsCoeff5',32);
            [app.c6, p6, DistHist6]=vqsplit(cepsCoeff6',32);
            [app.c7, p7, DistHist7]=vqsplit(cepsCoeff7',32);
            [app.c8, p8, DistHist8]=vqsplit(cepsCoeff8',32);
            [app.c9, p9, DistHist9]=vqsplit(cepsCoeff9',32);
            [app.c10, p10, DistHist10]=vqsplit(cepsCoeff10',32);
            [app.c11, p11, DistHist11]=vqsplit(cepsCoeff11',32);
        end
    end

    methods (Access = public)

        % Construct app
        function app = final_gui_exported

            % Create and configure components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end


%Local Functions------------------------------------
function mfcc_array = mfcc(s,fs,nfilt,norm,lift) % 'y' or normalization and 'l' for liftering
    if norm == 'y'
        s = normalize(s);
    end
    t1 = 0:1/fs:length(s)/fs-1/fs;
    %plot(t1,s),title('s1 audio input')
    N = 256;
    M = 100;
    n = 0:255;
    
    numFrames = floor((length(s)-N)/(N-M))+1;
    frames = zeros(numFrames,N);
    
    startindex = 1;
    endindex = N;
    w = hamming(N);
    y = zeros(numFrames,N);
    frames_fft = zeros(numFrames,N);
    P = zeros(numFrames,N);
    Pgram= zeros(numFrames,length(s));
    for k = 1:numFrames
        frames(k,:) = s(startindex:endindex);
        frames(k,:) = frames(k,:);
        startindex = startindex+N-M;
        endindex = startindex+N-1;
        % apply window
        y(k,:) = frames(k,:).*w';
        % compute FFT of each frame
        frames_fft(k,:) = fft(y(k,:));
        % compute periodogram
        P(k,:) = ((abs(exp(-1i.*M.*n.*(k-1)).*frames_fft(k,:))).^2)/N;
        zp = zeros(1,(k-1).*M);
        Pgram(k,:) = [zp P(k,:) zeros(1,abs(length(s)-length(zp)-length(P(k,:))))];
    end
    Pgram_sum = sum(Pgram);
    %plot(abs(Pgram_sum))
    start_freq = 300;
    end_freq = fs/2;
    m_start = 2595.*log10(1+start_freq/700);
    m_end = 2595.*log10(1+end_freq./700);
    m = linspace(m_start,m_end,nfilt+2);
    f_Hz = 700*(10.^(m./2595)-1);
    f_bin = floor((2*N+1).*f_Hz/fs);
    
    % generate filter bank
    fbank = zeros(20,256);
    for mel_k = 1:nfilt
        f_m_left = f_bin(mel_k);
        f_m_center = f_bin(mel_k+1);
        f_m_right = f_bin(mel_k+2);
        
        for k = f_m_left:f_m_center
            fbank(mel_k,k) = (k-f_bin(mel_k))/(f_bin(mel_k+1)-f_bin(mel_k));
            if fbank(mel_k,k)<=0
                fbank(mel_k,k)=0;
            end
        end
        for k = f_m_center:f_m_right
            fbank(mel_k,k) = (f_bin(mel_k+2)-k)/(f_bin(mel_k+2)-f_bin(mel_k+1));
            if fbank(mel_k,k)<=0
                fbank(mel_k,k)=0;
            end
        end
    end
    %plot(n,fbank)
    mfcc_array = zeros(numFrames,nfilt);
    for k = 1:numFrames
        mfcc_array(k,:) = dct(10*log10(fbank*P(k,:)'));
    end 
    if lift == 'l'
        w=1+(15/2)*sin((1:nfilt)*pi/15);
        mfcc_array = w.*mfcc_array;
    end
    %surf(-1*mfcc_array,'EdgeColor','none');   
    %axis xy; axis tight; colormap default; view(0,90); colorbar;
end

function [m, p, DistHist]=vqsplit(X,L)%--------------------------------------
% Vector Quantization: K-Means Algorithm with Spliting Method for Training
% NOT TESTED FOR CODEBOOK SIZES OTHER THAN POWERS OF BASE 2, E.G. 256, 512, ETC
% (Saves output to a mat file (CBTEMP.MAT) after each itteration, so that if
% it is going too slow you can break it (CTRL+C) without losing your work
% so far.)
% [M, P, DH]=VQSPLIT(X,L)
% 
% or
% [M_New, P, DH]=VQSPLIT(X,M_Old)   In this case M_Old is a codebook and is
%                                   retrained on data X
% 
% inputs:
% X: a matrix each column of which is a data vector
% L: codebook size (preferably a power of 2 e.g. 16,32 256, 1024) (Never
% tested for other values!
% 
% Outputs:
% M: the codebook as the centroids of the clusters
% P: Weight of each cluster the number of its vectors divided by total
%       number of vectors
% DH: The total distortion history, a vector containing the overall
% distortion of each itteration
%
% Method:
% The mean vector is split to two. the model is trained on those two vectors
% until the distortion does not vary much, then those are split to two and
% so on. until the disired number of clusters is reached.
% Algorithm:
% 1. Find the Mean
% 2. Split each centroid to two
% 3. Assign Each Data to a centroid
% 4. Find the Centroids
% 5. Calculate The Total Distance
% 6. If the Distance has not changed much
%       if the number of Centroids is smaller than L2 Goto Step 2
%       else Goto 7
%    Else (the Distance has changed substantialy) Goto Step 3
% 7. If the number of Centroids is larger than L
%    Discard the Centroid with (highest distortion OR lowest population)
%    Goto 3
% 8. Calculate the Variances and Cluster Weights if required
% 9. End
%
% Esfandiar Zavarehei, Brunel University
% May-2006
e=.01; % X---> [X-e*X and X+e*X] Percentage for Spliting
eRed=0.75; % Rate of reduction of split size, e, after each spliting. i.e. e=e*eRed;
DT=.005; % The threshold in improvement in Distortion before terminating and spliting again
DTRed=0.75; % Rate of reduction of Improvement Threshold, DT, after each spliting
MinPop=0.10; % The population of each cluster should be at least 10 percent of its quota (N/LC)
             % Otherwise that codeword is replaced with another codeword
d=size(X,1); % Dimension
N=size(X,2); % Number of Data points
isFirstRound=1; % First Itteration after Spliting
if numel(L)==1
    M=mean(X,2); % Mean Vector
    CB=[M*(1+e) M*(1-e)]; % Split to two vectors
else
    CB=L; % If the codebook is passed to the function just train it
    L=size(CB,2);
    e=e*(eRed^fix(log2(L)));
    DT=DT*(DTRed^fix(log2(L)));
end
LC=size(CB,2); % Current size of the codebook
Iter=0;
Split=0;
IsThereABestCB=0;
maxIterInEachSize=20; % The maximum number of training itterations at each 
                      % codebook size (The codebook size starts from one 
                      % and increases thereafter)
EachSizeIterCounter=0;
while 1
    %Distance Calculation
    [minIndx, dst]=VQIndex(X,CB); % Find the closest codewords to each data vector
    ClusterD=zeros(1,LC);
    Population=zeros(1,LC);
    LowPop=[];
    % Find the Centroids (Mean of each Cluster)
    for i=1:LC
        Ind=find(minIndx==i);
        if length(Ind)<MinPop*N/LC % if a cluster has very low population just remember it
            LowPop=[LowPop i];
        else
            CB(:,i)=mean(X(:,Ind),2);
            Population(i)=length(Ind);
            ClusterD(i)=sum(dst(Ind));
        end        
    end
    if ~isempty(LowPop)
        [temp MaxInd]=maxn(Population,length(LowPop));
        CB(:,LowPop)=CB(:,MaxInd)*(1+e); % Replace low-population codewords with splits of high population codewords
        CB(:,MaxInd)=CB(:,MaxInd)*(1-e);
        
        %re-train
        [minIndx, dst]=VQIndex(X,CB);
        ClusterD=zeros(1,LC);
        Population=zeros(1,LC);
        
        for i=1:LC
            Ind=find(minIndx==i);
            if ~isempty(Ind)
                CB(:,i)=mean(X(:,Ind),2);
                Population(i)=length(Ind);
                ClusterD(i)=sum(dst(Ind));
            else %if no vector is close enough to this codeword, replace it with a random vector
                CB(:,i)=X(:,fix(rand*N)+1);
                disp('A random vector was assigned as a codeword.')
                isFirstRound=1;% At least another iteration is required
            end                
        end
    end
    Iter=Iter+1;
    if isFirstRound % First itteration after a split (dont exit)
        TotalDist=sum(ClusterD(~isnan(ClusterD)));
        DistHist(Iter)=TotalDist;
        PrevTotalDist=TotalDist;        
        isFirstRound=0;
    else
        TotalDist=sum(ClusterD(~isnan(ClusterD)));  
        DistHist(Iter)=TotalDist;
        PercentageImprovement=((PrevTotalDist-TotalDist)/PrevTotalDist);
        if PercentageImprovement>=DT %Improvement substantial
            PrevTotalDist=TotalDist; %Save Distortion of this iteration and continue training
            isFirstRound=0;
        else%Improvement NOT substantial (Saturation)
            EachSizeIterCounter=0;
            if LC>=L %Enough Codewords?
                if L==LC %Exact number of codewords
                    disp(TotalDist)
                    break
                else % Kill one codeword at a time
                    [temp, Ind]=min(Population); % Eliminate low population codewords
                    NCB=zeros(d,LC-1);
                    NCB=CB(:,setxor(1:LC,Ind(1)));
                    CB=NCB;
                    LC=LC-1;
                    isFirstRound=1;
                end
            else %If not enough codewords yet, then Split more
                CB=[CB*(1+e) CB*(1-e)];
                e=eRed*e; %Split size reduction
                DT=DT*DTRed; %Improvement Threshold Reduction
                LC=size(CB,2);
                isFirstRound=1;
                Split=Split+1;
                IsThereABestCB=0; % As we just split this codebook, there is no best codebook at this size yet
                disp(LC)
            end
        end
    end    
    if ~IsThereABestCB
        BestCB=CB;
        BestD=TotalDist;
        IsThereABestCB=1;
    else % If there is a best CB, check to see if the current one is better than that
        if TotalDist<BestD
            BestCB=CB;
            BestD=TotalDist;
        end
    end
    EachSizeIterCounter=EachSizeIterCounter+1;
    if EachSizeIterCounter>maxIterInEachSize % If too many itterations in this size, stop training this size
        EachSizeIterCounter=0;
        CB=BestCB; % choose the best codebook so far
        IsThereABestCB=0;
        if LC>=L %Enough Codewords?
            if L==LC %Exact number of codewords
                disp(TotalDist)
                break
            else % Kill one codeword at a time
                [temp, Ind]=min(Population);
                NCB=zeros(d,LC-1);
                NCB=CB(:,setxor(1:LC,Ind(1)));
                CB=NCB;
                LC=LC-1;
                isFirstRound=1;
            end
        else %Split
            CB=[CB*(1+e) CB*(1-e)];
            e=eRed*e; %Split size reduction
            DT=DT*DTRed; %Improvement Threshold Reduction
            LC=size(CB,2);
            isFirstRound=1;
            Split=Split+1;
            IsThereABestCB=0;
            disp(LC)
        end
    end        
    disp(TotalDist)
    p=Population/N;
    save CBTemp CB p DistHist
end
m=CB;
p=Population/N;
disp(['Iterations = ' num2str(Iter)])
disp(['Split = ' num2str(Split)])

end


function [v, i]=maxn(x,n)%-------------------------------------------------------
% [V, I]=MAXN(X,N)
% APPLY TO VECTORS ONLY!
% This function returns the N maximum values of vector X with their indices.
% V is a vector which has the maximum values, and I is the index matrix,
% i.e. the indices corresponding to the N maximum values in the vector X
if nargin<2
    [v, i]=max(x); %Only the first maximum (default n=1)
else
    n=min(length(x),n);
    [v, i]=sort(x);
    v=v(end:-1:end-n+1);
    i=i(end:-1:end-n+1);    
end
end

function [I, dst]=VQIndex(X,CB)%------------------------------------------------- 
% Distance function
% Returns the closest index of vectors in X to codewords in CB
% In other words:
% I is a vector. The length of I is equal to the number of columns in X.
% Each element of I is the index of closest codeword (column) of CB to
% coresponding column of X
L=size(CB,2);
N=size(X,2);
LNThreshold=64*10000;
if L*N<LNThreshold
    D=zeros(L,N);
    for i=1:L
        D(i,:)=sum((repmat(CB(:,i),1,N)-X).^2,1);
    end
    [dst I]=min(D);
else
    I=zeros(1,N);
    dst=I;
    for i=1:N
        D=sum((repmat(X(:,i),1,L)-CB).^2,1);
        [dst(i) I(i)]=min(D);
    end
end
end

function [I, dist]=VQLSFSpectralIndex(X,CB,W)%--------------------------------------
% If your codewords are LSF coefficients, You can use this function instead of VQINDEX
% This is for speech coding
% I=VQLSFSPECTRALINDEX(X,CB,W)
% Calculates the nearest set of LSF coefficients in the codebook CB to each
% column of X by calculating their LP spectral distances.
% I is the index of the closest codeword, X is the set of LSF coefficients
% (each column is a set of coefficients) CB is the codebook, W is the
% weighting vector, if not provided it is assumed to be equal to ones(256,1)
% Esfandiar Zavarehei
% 9-Oct-05
if nargin<3
    L=256;
    W=ones(L,1);
else
    if isscalar(W)
        L=W;
        W=ones(L,1);
    elseif isvector(W)
        W=W(:);
        L=length(W);
    else
        error('Invalid input argument. W should be either a vector or a scaler!')
    end
end
NX=size(X,2);
NCB=size(CB,2);
AX=lsf2lpc(X);
ACB=lsf2lpc(CB);
D=zeros(NCB,1);
w=linspace(0,pi,L+1);
w=w(1:end-1);
N=size(AX,2)-1;
WFZ=zeros(N+1,L);
IMAGUNIT=sqrt(-1);
for k=0:N
    WFZ(k+1,:)=exp(IMAGUNIT*k*w);
end
SCB=zeros(L,NCB);
for i=1:NCB
    SCB(:,i)=(1./abs(ACB(i,:)*WFZ));
end
I=zeros(1,NX);
dist=zeros(1,NX);
for j=1:NX
    SX=(1./abs(AX(j,:)*WFZ))';    
    for i=1:NCB
        D(i)=sqrt(sum(((SX-SCB(:,i)).^2).*W));
    end
    [dist(j), I(j)]=min(D);
end
end


function [values] = similarityToAllCodebooks(cepsCoeff,frameNumber,melFiltersNum,codebookNum,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11)
    
   values(1) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c1,codebookNum);
   values(2) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c2,codebookNum);
   values(3) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c3,codebookNum);
   values(4) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c4,codebookNum);
   values(5) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c5,codebookNum);
   values(6) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c6,codebookNum);
   values(7) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c7,codebookNum);
   values(8) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c8,codebookNum);
   values(9) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c9,codebookNum);
   values(10) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c10,codebookNum);
   values(11) =  similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,c11,codebookNum);
end

function [score] = similarityToCodebook(cepsCoeff,frameNumber,melFiltersNum,codebook,codebookNum)
    score = 0;
    centroidCompare = whichCentroidToComputeDist(cepsCoeff,frameNumber,melFiltersNum,codebook,codebookNum);
    for mN = [1:melFiltersNum]
        for frames = [1:frameNumber]
           score = score + abs(cepsCoeff(frames,mN) - codebook(mN,centroidCompare(frames,mN)));
        end
    end
end

function [centroidCompare] = whichCentroidToComputeDist(cepsCoeff,frameNumber,melFiltersNum,codebook,codebookNum)
    for mN = [1:melFiltersNum]
        for frames = [1:frameNumber]
            for c = [1:codebookNum]
                temp(c) = abs(cepsCoeff(frames,mN) - codebook(mN,c));
            end
            [x,i] = min(temp);
            centroidCompare(frames,mN) = i;
        end
    end
end