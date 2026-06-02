"""Generate clean pedagogical diagrams for the lesson PDF."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

os.makedirs('figures', exist_ok=True)
BLUE = '#2c5f8a'; RED = '#b5341f'; GREEN = '#2e7d32'; GRAY = '#555'

# ---------------------------------------------------------------
# 1. Forward vs inverse schematic
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                fc=fc, ec='k', lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)
box(0.4, 2.4, 2.2, 1.1, "image $\\sigma$\n(784 numbers)", '#dce8f2')
box(4.1, 2.4, 1.6, 1.1, "$F$\n(simulator)", '#f2e3c6')
box(7.2, 2.4, 2.4, 1.1, "readings $y$\n(1900 numbers)", '#dce8f2')
ax.add_patch(FancyArrowPatch((2.7, 2.95), (4.0, 2.95), arrowstyle='-|>',
            mutation_scale=18, color=GREEN, lw=2))
ax.add_patch(FancyArrowPatch((5.8, 2.95), (7.1, 2.95), arrowstyle='-|>',
            mutation_scale=18, color=GREEN, lw=2))
ax.text(5, 3.6, "FORWARD (easy): image $\\rightarrow$ readings", ha='center',
        color=GREEN, fontsize=11, weight='bold')
ax.add_patch(FancyArrowPatch((7.1, 1.9), (2.7, 1.9), arrowstyle='-|>',
            mutation_scale=18, color=RED, lw=2,
            connectionstyle="arc3,rad=0.25"))
ax.text(5, 0.7, "INVERSE (hard): readings $\\rightarrow$ image\n"
        "(we are given $y$, must find $\\sigma$)", ha='center', color=RED,
        fontsize=11, weight='bold')
plt.tight_layout(); plt.savefig('figures/fig_forward_inverse.png', dpi=130,
        bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------
# 2. Bell curve + the likelihood guessing game
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4))
e = np.linspace(-4, 4, 400)
bell = np.exp(-e**2/2)
ax.plot(e, bell, color=BLUE, lw=2.5)
ax.fill_between(e, bell, color=BLUE, alpha=0.08)
# mark three candidates
pts = [(0.2, "guess 7\n(needs wobble 0.2)", GREEN),
       (2.2, "guess 6\n(needs wobble 2.2)", '#c79a00')]
for x, lab, c in pts:
    y = np.exp(-x**2/2)
    ax.plot([x, x], [0, y], color=c, lw=2, ls='--')
    ax.plot(x, y, 'o', color=c, ms=9)
    ax.annotate(lab, (x, y), xytext=(x+0.3, y+0.12), fontsize=10, color=c)
ax.annotate("guess 100 needs wobble $-185.8$:\nway off-screen, height $\\approx 0$",
            (3.6, 0.02), xytext=(1.2, 0.55), fontsize=10, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED))
ax.set_xlabel("size of the wobble (needed noise) $e$")
ax.set_ylabel("believability (height of bell)")
ax.set_title("The likelihood game: small wobble = believable = high score")
ax.set_ylim(0, 1.15); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_likelihood.png', dpi=130,
        bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------
# 3. Multimodal landscape: many valleys, gradient descent gets stuck
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.4))
x = np.linspace(0, 12, 800)
# three clear valleys (Gaussian dips) of similar depth on a gentle base
def dip(c, w, d): return -d*np.exp(-(x-c)**2/w)
land = 1.0 + 0.02*(x-6)**2 + dip(2.5,0.8,1.4) + dip(6.0,0.9,1.5) + dip(9.6,0.8,1.45)
ax.plot(x, land, color=GRAY, lw=2.4)
valleys = {2.5:'"2"', 6.0:'"5"', 9.6:'"9"'}
for vx, lab in valleys.items():
    vy = land[np.argmin(np.abs(x-vx))]
    ax.plot(vx, vy, 'o', color=BLUE, ms=8)
    # put the middle (deepest) label to the side so it clears the x-axis label
    if abs(vx-6.0) < 0.1:
        ax.text(vx+0.95, vy+0.05, "valley = "+lab, ha='left', fontsize=10,
                color=BLUE)
    else:
        ax.text(vx, vy-0.42, "valley = "+lab, ha='center', fontsize=10,
                color=BLUE)

def yat(xx): return land[np.argmin(np.abs(x-xx))]
# start A on the LEFT slope of the "2" valley -> rolls right-down into "2"
sA = 1.3; ax.plot(sA, yat(sA), 'o', color=GREEN, ms=11)
ax.text(sA, yat(sA)+0.28, "start A", ha='center', color=GREEN, weight='bold')
ax.annotate("", xy=(2.3, yat(2.3)), xytext=(1.5, yat(1.5)),
            arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2))
# start B on the LEFT slope of the "9" valley -> rolls right-down into "9"
sB = 8.6; ax.plot(sB, yat(sB), 'o', color=RED, ms=11)
ax.text(sB, yat(sB)+0.28, "start B", ha='center', color=RED, weight='bold')
ax.annotate("", xy=(9.4, yat(9.4)), xytext=(8.8, yat(8.8)),
            arrowprops=dict(arrowstyle='-|>', color=RED, lw=2))
ax.text(2.5, yat(2.5)+1.0, "A rolls\ninto \"2\"", ha='center', color=GREEN,
        fontsize=9)
ax.text(9.6, yat(9.6)+1.0, "B rolls\ninto \"9\"", ha='center', color=RED,
        fontsize=9)
ax.set_title("Many valleys (multimodality): downhill-walking only reaches the "
             "valley nearest its start")
ax.set_xlabel("possible images  $\\longrightarrow$")
ax.set_ylabel("how bad (lower = better)")
ax.set_yticks([]); ax.set_ylim(top=yat(0)+0.4)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_valleys.png', dpi=130,
        bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------
# 4. Flat sheet (PCA) vs curved surface (real digits / VAE)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
xc = np.linspace(1, 9, 400)
yc = 5 + 1.6*np.sin(0.9*xc) + 0.5*np.cos(2.1*xc)   # open wavy curve
def curve_y(xx): return 5 + 1.6*np.sin(0.9*xx) + 0.5*np.cos(2.1*xx)
true_x = 5.0; true = (true_x, curve_y(true_x))
for ax, mode in zip(axes, ['PCA (flat sheet)', 'VAE (curved sheet)']):
    ax.plot(xc, yc, color=BLUE, lw=2.6, label='surface of REAL digits')
    for tx in np.linspace(1.5, 8.5, 7):
        ax.plot(tx, curve_y(tx), 'o', color=BLUE, ms=5, alpha=0.55)
    ax.plot(*true, '*', color=GREEN, ms=22, label='the TRUE digit', zorder=5)
    if 'flat' in mode:
        lx = np.linspace(1.2, 8.8, 50); ly = 0.18*(lx-5)+5.1   # straight line
        ax.plot(lx, ly, '--', color=RED, lw=2.2, label='PCA flat sheet')
        # closest point on the flat line to the true digit (a WRONG digit)
        wx = 6.6; ax.plot(wx, 0.18*(wx-5)+5.1, 'X', color=RED, ms=15,
                label='PCA best = WRONG digit', zorder=5)
        ax.annotate("true digit lies\nOFF the flat sheet", true,
                    xytext=(1.3, 8.2), fontsize=9, color=GREEN,
                    arrowprops=dict(arrowstyle='->', color=GREEN))
        ax.annotate("flat sheet's closest\npoint = a different digit",
                    (wx,0.18*(wx-5)+5.1), xytext=(5.4, 2.4), fontsize=9,
                    color=RED, arrowprops=dict(arrowstyle='->', color=RED))
    else:
        ax.plot(*true, 'o', color=RED, ms=12, mfc='none', mew=2.5,
                label='VAE lands ON the true digit', zorder=6)
        ax.annotate("curved sheet passes\nTHROUGH the true digit", true,
                    xytext=(1.3, 8.2), fontsize=9, color=RED,
                    arrowprops=dict(arrowstyle='->', color=RED))
    ax.set_title(mode); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0.5, 9.5); ax.set_ylim(2, 9.0)
    ax.legend(fontsize=7.5, loc='lower right')
fig.suptitle("Why PCA gets the wrong digit: a FLAT sheet can't follow the "
             "CURVED surface of real digits", fontsize=11)
plt.tight_layout(); plt.savefig('figures/fig_flat_vs_curved.png', dpi=130,
        bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------
# 5. Soft thresholding: L1 zeroes small stuff, L2 only shrinks
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
d = np.linspace(-3, 3, 400); lam = 1.0
l1 = np.sign(d)*np.maximum(np.abs(d)-lam, 0)       # soft threshold
l2 = d/(1+lam)                                     # ridge shrink
ax.plot(d, d, color=GRAY, ls=':', lw=1.5, label='no penalty (keep as-is)')
ax.plot(d, l1, color=RED, lw=2.5, label='$\\ell_1$: zero out small, shift big')
ax.plot(d, l2, color=BLUE, lw=2.5, label='$\\ell_2$ (Tikhonov): just shrink')
ax.axvspan(-lam, lam, color=RED, alpha=0.08)
ax.text(0, -2.4, "inside this band\n$\\ell_1$ sets it to EXACTLY 0",
        ha='center', fontsize=9, color=RED)
ax.set_xlabel("what the data wants this pixel to be")
ax.set_ylabel("what the method actually outputs")
ax.set_title("Soft thresholding: why $\\ell_1$ gives clean (sparse) images")
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.2)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_softthreshold.png', dpi=130,
        bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------
# 6. Gradient descent: zigzag (no momentum) vs smooth (momentum)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
X, Y = np.meshgrid(np.linspace(-3,3,200), np.linspace(-2,2,200))
Z = 0.15*X**2 + Y**2          # elongated bowl (ill-conditioned)
for ax, mode in zip(axes, ['Plain descent (zig-zags)','With momentum (smooth)']):
    ax.contour(X, Y, Z, levels=18, colors='#bbb', linewidths=0.7)
    # simulate paths
    p = np.array([-2.7, 1.7]); path=[p.copy()]; v=np.zeros(2)
    for _ in range(40):
        g = np.array([0.3*p[0], 2*p[1]])
        if 'momentum' in mode:
            v = 0.8*v + 0.18*g; p = p - v
        else:
            p = p - 0.18*g
        path.append(p.copy())
    path=np.array(path)
    c = GREEN if 'momentum' in mode else RED
    ax.plot(path[:,0], path[:,1], '-o', color=c, ms=3, lw=1.3)
    ax.plot(0,0,'*',color='k',ms=14)
    ax.set_title(mode); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Walking downhill in a long narrow valley: momentum avoids the "
             "zig-zag", fontsize=11)
plt.tight_layout(); plt.savefig('figures/fig_gd_momentum.png', dpi=130,
        bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------
# 7. Template matching schematic
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis('off')
ax.set_xlim(0,10); ax.set_ylim(0,4)
ax.text(5, 3.7, "Template matching: simulate each of 60,000 real digits, "
        "keep the closest match", ha='center', fontsize=11, weight='bold')
for i,(d,match) in enumerate([("2",False),("5",False),("5",True),("8",False)]):
    x0 = 0.5 + i*2.4
    box_c = '#cdeccd' if match else '#eee'
    ax.add_patch(FancyBboxPatch((x0,1.4),1.0,1.0,boxstyle="round,pad=0.05",
                fc=box_c, ec='k'))
    ax.text(x0+0.5,1.9,d,ha='center',va='center',fontsize=20)
    ax.text(x0+0.5,1.15,"$F(\\cdot)$",ha='center',fontsize=9)
    err = ["err 8e-4","err 5e-4","err 2e-5\nBEST","err 3e-4"][i]
    ax.text(x0+0.5,0.5,err,ha='center',fontsize=8.5,
            color=GREEN if match else GRAY,
            weight='bold' if match else 'normal')
    ax.text(x0+1.5,1.9,"vs $y_{obs}$",ha='center',fontsize=8,color=GRAY)
ax.text(9.4,1.9,"...",fontsize=16)
plt.tight_layout(); plt.savefig('figures/fig_template_match.png', dpi=130,
        bbox_inches='tight'); plt.close()

print("Generated 7 teaching figures in figures/")
