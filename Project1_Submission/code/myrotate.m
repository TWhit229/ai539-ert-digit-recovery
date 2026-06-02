function out = myrotate(img, theta_deg)
%MYROTATE  Bilinear CCW rotation of img by theta_deg around its center,
%cropped to the original size, with zero fill outside. A no-toolbox
%replacement for imrotate(img, theta_deg, 'bilinear', 'crop').

[H, W] = size(img);
cx = (W + 1) / 2;
cy = (H + 1) / 2;
th = theta_deg * pi / 180;

[X, Y] = meshgrid(1:W, 1:H);
Xc = X - cx; Yc = Y - cy;

% inverse rotation: for each output pixel, find its source pixel in the input
Xs =  cos(th) * Xc + sin(th) * Yc + cx;
Ys = -sin(th) * Xc + cos(th) * Yc + cy;

out = interp2(img, Xs, Ys, 'linear', 0);
end
