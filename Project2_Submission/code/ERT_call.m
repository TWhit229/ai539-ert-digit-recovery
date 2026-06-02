function [y, J] = ERT_call(sig, evalMode, gridSize, sigmaBg, Lx, Ly)
%ERT_CALL Thin wrapper around ERT2D that caches ERTParams on the MATLAB side,
%so Python callers never have to marshal the sparse matrices inside ERTParams.
%
%  Inputs:
%    sig       — conductivity image (gridSize_y x gridSize_x), full/dense
%    evalMode  — 1 (forward only) or 2 (forward + Jacobian)
%    gridSize  — [ny nx]  (defaults [28 28])
%    sigmaBg   — background conductivity (default 1)
%    Lx, Ly    — domain dimensions (defaults 10, 5)
%
%  Outputs:
%    y — sens_data, dense vector of length 1900
%    J — Jacobian (only if evalMode == 2). Returns [] for evalMode == 1.
%
%  Cached state: ERTParams (recomputed only when gridSize changes).

persistent ERTParams cachedSize cachedBg cachedLx cachedLy

if isempty(ERTParams) ...
        || ~isequal(size(cachedSize), size(gridSize)) ...
        || any(cachedSize ~= gridSize) ...
        || cachedBg ~= sigmaBg ...
        || cachedLx ~= Lx ...
        || cachedLy ~= Ly
    ERTParams  = paramPackGenerator(gridSize, sigmaBg, Lx, Ly);
    cachedSize = gridSize;
    cachedBg   = sigmaBg;
    cachedLx   = Lx;
    cachedLy   = Ly;
end

if evalMode == 1
    [y, ~] = ERT2D(sig, 1, ERTParams);
    J = [];
else
    [y, J] = ERT2D(sig, 2, ERTParams);
end

y = full(y);
if ~isempty(J), J = full(J); end
end
