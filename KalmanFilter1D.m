clear; clc; close all;

% init params A, H, Q, R
% init + sim gt and measurement z (random walk model) vectors 
% init Xhat, XPred vectors, state covariance (initial filter guess)
% set first time step of Xhat to 0. Ignore setting Xpred as first element never used
% kalman filter loop (prediction + estimation)
% plot results
% This sim purposely not computationally efficient (ie frequest use & transforming of scalar variables).

nSteps = 100; % time step for simulation
A = 1; % state transition matrix = scalar due to influence directed at one state variable (position)
H = 1; % measurement matrix = scalar due to linear state->measurement space transform
Q = 0.000001; % process noise (kalman model)
R = 0.01;  % sensor noise (measurement)

groundTruthMat = zeros(1, nSteps); % init ground truth with zeros 

for k = 2:nSteps % linear recursive func defining ground truth
    w = sqrt(Q) * randn(); % sim different noise at each time step (avoid linearly increasing ground truth)
    groundTruthMat(k) = (A * groundTruthMat(k - 1)) + w; % ground truth val at k time step
end

sensorMat = (H * groundTruthMat) + (sqrt(R) * randn(1, nSteps)); % measurement data model

xHat = zeros(1, nSteps); 
xPred = zeros(1, nSteps);
P = 1.0; % initial state-space guess for filter
xHat(1) = 0;

for i = 2:nSteps

    % Prediction step
    xPred(i) = A * xHat(i - 1);
    P = (A * P * A) + Q;

    % Estimation Step
    K = P * transpose(H) / (H * P * transpose(H) + R);
    xHat(i) = xPred(i) + K * (sensorMat(i) - H * xPred(i));
    P = P * (1 - K * H);

end

t = 1:nSteps;

figure;
hold on; grid on;
plot(t, groundTruthMat, 'b-', 'LineWidth', 1.5);
plot(t, sensorMat, 'k.', 'MarkerSize', 10);
plot(t, xHat, 'r-', 'LineWidth', 1.5);
xlabel('Time step');
ylabel('State value');
title('1-D Scalar Kalman Filter (using random walk model');
legend('True state', 'Measurements', 'Kalman estimate', 'Location', 'best');
hold off;



