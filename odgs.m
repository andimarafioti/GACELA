
base_folder = '\\kfsnas08\Denklast\amarafioti\Documents\testGAN\stftGAN\GANinpainting\fma_electronic\';

pghi_folder = 'pghi\';
real_folder = 'real\';
clicked_folder = 'clicked\';
GAN_folder = 'GAN_400k\';

num_data = 64;
num_methods = 4;
ODG = zeros(num_data,num_methods-1);

for index = 1:num_data
    [real, fs] =audioread(strcat(base_folder, real_folder, num2str(index-1), '.wav'));
    real_resampled = resample(double(real), 48000, fs);
    audiowrite('real_48000.wav', real_resampled, 48000)
    

    [pghi, fs] =audioread(strcat(base_folder, pghi_folder, num2str(index-1), '.wav'));
    pghi_resampled = resample(double(pghi), 48000, fs);
    audiowrite('pghi_48000.wav', pghi_resampled, 48000)

 
    [odg, movb] = PQevalAudio_fn('real_48000.wav', 'pghi_48000.wav');
    ODG(index, 1) = odg;
 
    [click, fs] =audioread(strcat(base_folder, clicked_folder, num2str(index-1), '.wav'));
    click_resampled = resample(double(click), 48000, fs);
    audiowrite('click_48000.wav', click_resampled, 48000)
    
    [odg, movb] = PQevalAudio_fn('real_48000.wav', 'click_48000.wav');
    ODG(index, 2) = odg;
    
    [GAN, fs] =audioread(strcat(base_folder, GAN_folder, num2str(index-1), '.wav'));
    GAN_resampled = resample(double(GAN), 48000, fs);
    audiowrite('GAN_48000.wav', GAN_resampled, 48000)
    
    [odg, movb] = PQevalAudio_fn('real_48000.wav', 'GAN_48000.wav');
    ODG(index, 3) = odg;
    
end