"""
Generate four technical diagrams for non-cooperative satellite grasping:
1. Base image: Satellites with robotic arm (chaser arm reaching toward target)
2. Geometric cues: Point cloud, wireframe, bounding box, feature edges
3. Physical cues: Mass distribution, CoM, inertia axes, surface normals, reflectance zones
4. State cues: Tumbling angular velocity, trajectory prediction, attitude quaternion visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch, Circle, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.collections import LineCollection
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def dark_bg(fig, ax):
    fig.patch.set_facecolor('#05060f')
    ax.set_facecolor('#05060f')

def star_field(ax, n=350, xlim=(-1, 1), ylim=(-1, 1)):
    rng = np.random.default_rng(42)
    xs = rng.uniform(*xlim, n)
    ys = rng.uniform(*ylim, n)
    sizes = rng.uniform(0.3, 2.5, n)
    ax.scatter(xs, ys, s=sizes, c='white', alpha=0.6, zorder=0)

def draw_solar_panel(ax, cx, cy, angle_deg, width=0.22, height=0.09, color='#1a3a6e', grid_color='#4a8fdf'):
    """Draw a solar panel rectangle with grid lines."""
    from matplotlib.transforms import Affine2D
    panel = FancyBboxPatch(
        (-width/2, -height/2), width, height,
        boxstyle="round,pad=0.005",
        facecolor=color, edgecolor=grid_color, linewidth=0.8,
        transform=ax.transData, zorder=4
    )
    t = (transforms.Affine2D().rotate_deg(angle_deg).translate(cx, cy) + ax.transData)
    panel.set_transform(t)
    ax.add_patch(panel)
    # grid lines
    for frac in [0.33, 0.66]:
        xg = -width/2 + frac*width
        line = plt.Line2D([xg, xg], [-height/2, height/2], color=grid_color, lw=0.5,
                          transform=t, zorder=5)
        ax.add_line(line)
    for frac in [0.5]:
        yg = -height/2 + frac*height
        line = plt.Line2D([-width/2, width/2], [yg, yg], color=grid_color, lw=0.5,
                          transform=t, zorder=5)
        ax.add_line(line)

def draw_cuboid(ax, cx, cy, w, h, body_color, edge_color, depth_color=None, zorder=3):
    """Draw a simple isometric-style cuboid."""
    if depth_color is None:
        depth_color = '#222'
    dx, dy = 0.06, 0.04
    # front face
    front = plt.Polygon([[cx-w/2, cy-h/2],[cx+w/2, cy-h/2],[cx+w/2, cy+h/2],[cx-w/2, cy+h/2]],
                         closed=True, facecolor=body_color, edgecolor=edge_color, linewidth=0.9, zorder=zorder)
    # top face
    top = plt.Polygon([[cx-w/2, cy+h/2],[cx+w/2, cy+h/2],[cx+w/2+dx, cy+h/2+dy],[cx-w/2+dx, cy+h/2+dy]],
                       closed=True, facecolor=depth_color, edgecolor=edge_color, linewidth=0.9, zorder=zorder)
    # right face
    right = plt.Polygon([[cx+w/2, cy-h/2],[cx+w/2+dx, cy-h/2+dy],[cx+w/2+dx, cy+h/2+dy],[cx+w/2, cy+h/2]],
                         closed=True, facecolor=depth_color, edgecolor=edge_color, linewidth=0.9, zorder=zorder)
    for p in [front, top, right]:
        ax.add_patch(p)

# ─────────────────────────────────────────────────────────────
# IMAGE 0 — CHASER + TARGET + ROBOTIC ARM
# ─────────────────────────────────────────────────────────────

def draw_robotic_arm_base():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    dark_bg(fig, ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    star_field(ax, n=500, xlim=(-1,1), ylim=(-0.6,0.6))

    # ── Earth glow bottom-right ──
    from matplotlib.patches import Ellipse
    earth = Ellipse((0.88, -0.45), 0.55, 0.38,
                    facecolor='#1a4a7a', edgecolor='#3a8fbf', linewidth=1.5, alpha=0.55, zorder=1)
    ax.add_patch(earth)
    earth2 = Ellipse((0.88, -0.45), 0.5, 0.33,
                     facecolor='#2060a0', edgecolor='none', alpha=0.3, zorder=1)
    ax.add_patch(earth2)
    ax.text(0.88, -0.55, 'Earth', color='#88ccff', fontsize=7, ha='center', va='top', alpha=0.7, zorder=2)

    # ══════════════════════════════════
    # CHASER spacecraft (left, golden)
    # ══════════════════════════════════
    cx, cy = -0.48, 0.05
    # Main body
    draw_cuboid(ax, cx, cy, 0.18, 0.16, '#c8a240', '#e8c860', depth_color='#6a5010', zorder=5)
    # Side solar panels
    draw_solar_panel(ax, cx - 0.22, cy, 0, width=0.20, height=0.09)
    draw_solar_panel(ax, cx + 0.22, cy, 0, width=0.20, height=0.09)
    # Top panel (small)
    draw_solar_panel(ax, cx, cy + 0.16, 90, width=0.08, height=0.06)
    # Label
    ax.text(cx, cy - 0.16, 'Chaser S/C', color='#ffd060', fontsize=9, ha='center', va='top',
            fontweight='bold', zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground='#05060f')])

    # ══════════════════════════════════
    # TARGET spacecraft (right, dark/silver — non-cooperative)
    # ══════════════════════════════════
    tx, ty = 0.45, -0.02
    draw_cuboid(ax, tx, ty, 0.22, 0.18, '#3a3f4e', '#8090b0', depth_color='#1a2030', zorder=5)
    # Solar panels (one deployed, one partially folded)
    draw_solar_panel(ax, tx - 0.26, ty + 0.03, -8, width=0.22, height=0.10, color='#1a2a4a', grid_color='#5080c0')
    draw_solar_panel(ax, tx + 0.27, ty - 0.02, 5, width=0.22, height=0.10, color='#1a2a4a', grid_color='#5080c0')
    # Antenna dish
    dish = Arc((tx+0.05, ty+0.13), 0.10, 0.07, angle=0, theta1=180, theta2=360,
               color='#b0c0d0', linewidth=1.2, zorder=6)
    ax.add_patch(dish)
    ax.plot([tx+0.05, tx+0.05], [ty+0.09, ty+0.13], color='#b0c0d0', lw=0.8, zorder=6)
    # Tumble indicator arrows (slightly tilted)
    rot_arrow = FancyArrowPatch((tx+0.28, ty+0.13), (tx+0.18, ty+0.22),
                                 arrowstyle='->', color='#ff9040', lw=1.2,
                                 connectionstyle='arc3,rad=0.4', zorder=8)
    ax.add_patch(rot_arrow)
    ax.text(tx+0.34, ty+0.20, 'tumbling', color='#ff9040', fontsize=7, ha='left', va='center',
            style='italic', zorder=10,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#05060f')])
    ax.text(tx, ty - 0.17, 'Target S/C (non-cooperative)', color='#90a8c8', fontsize=9,
            ha='center', va='top', zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground='#05060f')])

    # ══════════════════════════════════
    # ROBOTIC ARM (7-DOF snake from chaser to target)
    # ══════════════════════════════════
    # Joint positions (arm extends from chaser right side toward target)
    j0 = np.array([cx + 0.09, cy + 0.04])   # shoulder mount on chaser
    j1 = np.array([cx + 0.22, cy + 0.15])
    j2 = np.array([cx + 0.34, cy + 0.22])
    j3 = np.array([cx + 0.48, cy + 0.18])
    j4 = np.array([cx + 0.60, cy + 0.10])
    j5 = np.array([cx + 0.72, cy + 0.04])
    j6 = np.array([cx + 0.84, cy - 0.02])
    # Gripper tips toward target docking port
    grip_base = np.array([tx - 0.14, ty + 0.02])
    finger1   = np.array([tx - 0.09, ty + 0.06])
    finger2   = np.array([tx - 0.09, ty - 0.02])

    joints = [j0, j1, j2, j3, j4, j5, j6, grip_base]
    arm_color = '#d0d8f0'
    joint_color = '#ffd060'

    # Draw links
    for i in range(len(joints)-1):
        ax.plot([joints[i][0], joints[i+1][0]],
                [joints[i][1], joints[i+1][1]],
                color=arm_color, lw=3.5, solid_capstyle='round', zorder=7)
        ax.plot([joints[i][0], joints[i+1][0]],
                [joints[i][1], joints[i+1][1]],
                color='#2a3050', lw=1.0, solid_capstyle='round', zorder=7, ls='--', alpha=0.5)

    # Draw joints
    for j in joints[1:-1]:
        ax.plot(*j, 'o', color=joint_color, ms=5, zorder=9)
        ax.plot(*j, 'o', color='#05060f', ms=2.5, zorder=10)

    # Gripper fingers
    ax.plot([grip_base[0], finger1[0]], [grip_base[1], finger1[1]],
            color='#ffd060', lw=2.5, solid_capstyle='round', zorder=9)
    ax.plot([grip_base[0], finger2[0]], [grip_base[1], finger2[1]],
            color='#ffd060', lw=2.5, solid_capstyle='round', zorder=9)
    # Gripper pads
    ax.plot(*finger1, 's', color='#ff8040', ms=4, zorder=10)
    ax.plot(*finger2, 's', color='#ff8040', ms=4, zorder=10)

    # Arm label
    ax.text((j3[0]+j4[0])/2, j3[1]+0.07, '7-DOF Robotic Arm',
            color='#d0d8f0', fontsize=8, ha='center', va='bottom',
            path_effects=[pe.withStroke(linewidth=2, foreground='#05060f')], zorder=11)
    # Arrow pointing at arm
    ax.annotate('', xy=(j3[0], j3[1]+0.015), xytext=(j3[0], j3[1]+0.06),
                arrowprops=dict(arrowstyle='->', color='#d0d8f0', lw=0.8))

    # Capture zone dashed circle around target grapple point
    cap_circ = Circle((tx-0.11, ty+0.02), 0.10, fill=False, edgecolor='#60ff80',
                       linewidth=1.2, linestyle='--', alpha=0.8, zorder=8)
    ax.add_patch(cap_circ)
    ax.text(tx-0.11, ty-0.11, 'Capture\nZone', color='#60ff80', fontsize=7, ha='center', va='top',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#05060f')], zorder=11)

    # Title
    ax.set_title('On-Orbit Servicing — Robotic Arm Capture Demonstration\n(Non-Cooperative Target)',
                 color='white', fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    fig.savefig('/workspace/output_0_robotic_arm.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved: output_0_robotic_arm.png")


# ─────────────────────────────────────────────────────────────
# IMAGE 1 — GEOMETRIC CUES
# ─────────────────────────────────────────────────────────────

def draw_geometric_cues():
    fig = plt.figure(figsize=(14, 9), dpi=150)
    fig.patch.set_facecolor('#05060f')

    # 2×3 grid of sub-panels
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.04, right=0.97, top=0.88, bottom=0.06)

    panel_titles = [
        'Edge Detection\n& Contour Lines',
        'Sparse Keypoint\nFeature Map',
        '3-D Point Cloud\n(LiDAR)',
        'Bounding Box &\nOBB Estimation',
        'Depth-Map\n(stereo / ToF)',
        'Wireframe CAD\nModel Matching',
    ]
    colors_accent = ['#40d0ff','#ffd040','#60ff80','#ff6060','#c060ff','#ff9040']

    rng = np.random.default_rng(7)

    for idx, (title, acc) in enumerate(zip(panel_titles, colors_accent)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0e1a')
        ax.set_aspect('equal')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a3050')

        # Star micro-field
        xs = rng.uniform(-1, 1, 80)
        ys = rng.uniform(-1, 1, 80)
        ax.scatter(xs, ys, s=0.5, c='white', alpha=0.4, zorder=0)

        # Satellite silhouette (simple polygon)
        body = np.array([[-0.22,-0.18],[0.22,-0.18],[0.22,0.18],[-0.22,0.18]])
        sp_l = np.array([[-0.60,-0.06],[-0.24,-0.06],[-0.24,0.06],[-0.60,0.06]])
        sp_r = np.array([[0.24,-0.06],[0.60,-0.06],[0.60,0.06],[0.24,0.06]])

        if idx == 0:  # Edge detection
            # Draw glowing contour edges
            for pts, lw in [(body,2.0),(sp_l,1.5),(sp_r,1.5)]:
                ax.fill(pts[:,0], pts[:,1], color='#1a2030', zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color=acc, lw=lw, zorder=4,
                        path_effects=[pe.withStroke(linewidth=lw+1.5, foreground=acc, alpha=0.3)])
            # Dashed edge lines on body
            ax.plot([-0.22, 0.22], [0, 0], color=acc, lw=0.6, ls='--', alpha=0.5, zorder=4)
            ax.plot([0, 0], [-0.18, 0.18], color=acc, lw=0.6, ls='--', alpha=0.5, zorder=4)
            ax.text(0, -0.85, 'Canny / Sobel gradient edges', color=acc, fontsize=7, ha='center')

        elif idx == 1:  # Keypoint feature map
            for pts in [body, sp_l, sp_r]:
                ax.fill(pts[:,0], pts[:,1], color='#1a2030', zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#3040a0', lw=0.8, zorder=3)
            # SIFT-like keypoints
            kps_x = np.array([-0.22, 0.22, 0.22, -0.22, 0, -0.60, 0.60, 0, -0.41, 0.41, -0.06])
            kps_y = np.array([-0.18,-0.18, 0.18, 0.18, 0, 0, 0, 0.18, 0, 0, -0.18])
            for kx, ky in zip(kps_x, kps_y):
                ax.plot(kx, ky, 'o', color=acc, ms=4, zorder=5)
                circ = Circle((kx,ky), 0.08+rng.uniform(0,0.04), fill=False,
                               edgecolor=acc, lw=0.6, alpha=0.5, zorder=5)
                ax.add_patch(circ)
                angle = rng.uniform(0, 2*np.pi)
                ax.annotate('', xy=(kx+0.09*np.cos(angle), ky+0.09*np.sin(angle)),
                            xytext=(kx, ky),
                            arrowprops=dict(arrowstyle='->', color=acc, lw=0.7), zorder=6)
            ax.text(0, -0.85, 'ORB / SIFT feature descriptors', color=acc, fontsize=7, ha='center')

        elif idx == 2:  # Point cloud
            for pts in [body, sp_l, sp_r]:
                ax.fill(pts[:,0], pts[:,1], color='#0a0f1f', zorder=2)
            # LiDAR point cloud sampled on satellite surface
            def sample_rect(lo_x, hi_x, lo_y, hi_y, n):
                xs = rng.uniform(lo_x, hi_x, n)
                ys = rng.uniform(lo_y, hi_y, n)
                return xs, ys
            px1,py1 = sample_rect(-0.22,0.22,-0.18,0.18, 320)
            px2,py2 = sample_rect(-0.60,-0.24,-0.06,0.06, 120)
            px3,py3 = sample_rect(0.24,0.60,-0.06,0.06, 120)
            all_x = np.concatenate([px1,px2,px3])
            all_y = np.concatenate([py1,py2,py3])
            dist = np.sqrt(all_x**2 + all_y**2)
            colors_pt = plt.cm.plasma(dist / dist.max())
            ax.scatter(all_x, all_y, s=1.2, c=colors_pt, alpha=0.85, zorder=4)
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(dist.min(), dist.max()))
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label('range (m)', color='white', fontsize=6)
            cbar.ax.yaxis.set_tick_params(color='white', labelsize=5, labelcolor='white')
            ax.text(0, -0.85, 'LiDAR / depth point cloud', color=acc, fontsize=7, ha='center')

        elif idx == 3:  # Bounding box
            for pts in [body, sp_l, sp_r]:
                ax.fill(pts[:,0], pts[:,1], color='#1a2030', zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#3040a0', lw=0.7, zorder=3)
            # AABB
            aabb = FancyBboxPatch((-0.62,-0.20),1.24,0.40,
                                  boxstyle='square,pad=0', fill=False,
                                  edgecolor=acc, linewidth=1.5, linestyle='--', zorder=5)
            ax.add_patch(aabb)
            ax.text(0.63, 0.21, 'AABB', color=acc, fontsize=7, va='bottom')
            # OBB (rotated)
            angle_obb = 12
            obb = FancyBboxPatch((-0.60,-0.19),1.20,0.38,
                                 boxstyle='square,pad=0', fill=False,
                                 edgecolor='#ff9040', linewidth=1.2,
                                 transform=(transforms.Affine2D().rotate_deg(angle_obb) + ax.transData),
                                 zorder=5)
            ax.add_patch(obb)
            ax.text(0.50, -0.42, 'OBB', color='#ff9040', fontsize=7)
            # Corner markers
            corners = [(-0.62,-0.20),(-0.62,0.20),(0.62,0.20),(0.62,-0.20)]
            for c in corners:
                ax.plot(*c, 's', color=acc, ms=4, zorder=6)
            ax.text(0, -0.85, 'AABB / OBB object bounding box', color=acc, fontsize=7, ha='center')

        elif idx == 4:  # Depth map
            for pts in [body, sp_l, sp_r]:
                ax.fill(pts[:,0], pts[:,1], color='#0a0f1f', zorder=2)
            # False-color depth image
            xx, yy = np.meshgrid(np.linspace(-0.7,0.7,200), np.linspace(-0.25,0.25,80))
            depth = np.zeros_like(xx)
            mask_body = (np.abs(xx)<0.22) & (np.abs(yy)<0.18)
            mask_sp_l = (xx>-0.60)&(xx<-0.24) & (np.abs(yy)<0.06)
            mask_sp_r = (xx>0.24)&(xx<0.60) & (np.abs(yy)<0.06)
            depth[mask_body] = 1.0 - 0.6*(xx[mask_body]**2 + yy[mask_body]**2)
            depth[mask_sp_l] = 0.5 - 0.3*np.abs(xx[mask_sp_l]+0.42)
            depth[mask_sp_r] = 0.5 - 0.3*np.abs(xx[mask_sp_r]-0.42)
            depth = np.clip(depth, 0, 1)
            im = ax.imshow(depth, extent=[-0.7,0.7,-0.25,0.25], cmap='RdYlBu_r',
                           vmin=0, vmax=1, alpha=0.85, zorder=3, aspect='auto')
            cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label('depth', color='white', fontsize=6)
            cb.ax.yaxis.set_tick_params(color='white', labelsize=5, labelcolor='white')
            ax.text(0, -0.85, 'Stereo / ToF depth image', color=acc, fontsize=7, ha='center')

        elif idx == 5:  # Wireframe matching
            for pts in [body, sp_l, sp_r]:
                ax.fill(pts[:,0], pts[:,1], color='#0a0a18', zorder=2)
            # Wireframe overlay lines
            wire_lines = [
                [(-0.22,-0.18),(0.22,-0.18)],
                [(0.22,-0.18),(0.22,0.18)],
                [(0.22,0.18),(-0.22,0.18)],
                [(-0.22,0.18),(-0.22,-0.18)],
                [(-0.22,-0.18),(0,0)],
                [(0.22,-0.18),(0,0)],
                [(0.22,0.18),(0,0)],
                [(-0.22,0.18),(0,0)],
                [(-0.60,-0.06),(-0.24,-0.06)],
                [(-0.24,-0.06),(-0.24,0.06)],
                [(-0.24,0.06),(-0.60,0.06)],
                [(-0.60,0.06),(-0.60,-0.06)],
                [(0.24,-0.06),(0.60,-0.06)],
                [(0.60,-0.06),(0.60,0.06)],
                [(0.60,0.06),(0.24,0.06)],
                [(0.24,0.06),(0.24,-0.06)],
                [(-0.60,-0.06),(-0.60-0.04,-0.06-0.04)],
                [(0.60,-0.06),(0.60+0.04,-0.06-0.04)],
            ]
            lc = LineCollection(wire_lines, colors=acc, linewidths=0.9, alpha=0.85, zorder=4)
            ax.add_collection(lc)
            # Residual error arrows (misalignment)
            offsets = [(0.22,-0.18,0.04,-0.03),(-0.22,0.18,-0.03,0.04),(0.22,0.18,0.03,0.04)]
            for ox,oy,dx,dy in offsets:
                ax.annotate('', xy=(ox+dx,oy+dy), xytext=(ox,oy),
                            arrowprops=dict(arrowstyle='->', color='#ff4040', lw=0.8), zorder=6)
            ax.text(0, -0.85, 'CAD wireframe pose alignment', color=acc, fontsize=7, ha='center')

        ax.set_title(title, color=acc, fontsize=9, fontweight='bold', pad=4)

    fig.suptitle('Geometric Cues for Non-Cooperative Target Grasping  |  几何线索',
                 color='white', fontsize=14, fontweight='bold', y=0.96)

    fig.savefig('/workspace/output_1_geometric_cues.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved: output_1_geometric_cues.png")


# ─────────────────────────────────────────────────────────────
# IMAGE 2 — PHYSICAL CUES
# ─────────────────────────────────────────────────────────────

def draw_physical_cues():
    fig = plt.figure(figsize=(14, 9), dpi=150)
    fig.patch.set_facecolor('#05060f')

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.04, right=0.97, top=0.88, bottom=0.06)

    panel_titles = [
        'Mass Distribution\n& CoM Estimation',
        'Inertia Tensor\nPrincipal Axes',
        'Surface Material\n& Reflectance',
        'Thermal Emission\nMap (IR)',
        'Solar Radiation\nPressure Force',
        'Structural Stiffness\n& Grapple Points',
    ]
    colors_accent = ['#ffaa40','#60d0ff','#ff60a0','#ff5040','#ffd040','#60ff80']
    rng = np.random.default_rng(13)

    for idx, (title, acc) in enumerate(zip(panel_titles, colors_accent)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0e1a')
        ax.set_aspect('equal')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a3050')

        xs = rng.uniform(-1, 1, 60); ys = rng.uniform(-1, 1, 60)
        ax.scatter(xs, ys, s=0.5, c='white', alpha=0.3, zorder=0)

        body_pts  = np.array([[-0.22,-0.18],[0.22,-0.18],[0.22,0.18],[-0.22,0.18]])
        sp_l_pts  = np.array([[-0.60,-0.06],[-0.24,-0.06],[-0.24,0.06],[-0.60,0.06]])
        sp_r_pts  = np.array([[0.24,-0.06],[0.60,-0.06],[0.60,0.06],[0.24,0.06]])

        if idx == 0:  # Mass distribution
            # Draw body with heatmap overlay simulating density
            ax.fill(body_pts[:,0], body_pts[:,1], color='#1a2030', zorder=2)
            for pts in [sp_l_pts, sp_r_pts]:
                ax.fill(pts[:,0], pts[:,1], color='#141e2e', zorder=2)
            # Density dots
            n_dense = 400
            bx = rng.uniform(-0.22,0.22,n_dense)
            by = rng.uniform(-0.18,0.18,n_dense)
            # Heavier near center/bottom (tanks)
            weight = np.exp(-2*(bx**2 + (by+0.08)**2))
            weight /= weight.sum()
            sel = rng.choice(n_dense, size=250, p=weight, replace=False)
            ax.scatter(bx[sel], by[sel], s=2.5, c=acc, alpha=0.7, zorder=4)
            ax.scatter(bx[~np.isin(np.arange(n_dense),sel)],
                       by[~np.isin(np.arange(n_dense),sel)], s=0.8, c='#3050a0', alpha=0.5, zorder=3)
            # CoM marker
            com_x, com_y = 0.02, -0.04
            ax.plot(com_x, com_y, '+', color='white', ms=12, mew=2, zorder=7)
            ax.plot(com_x, com_y, 'o', color=acc, ms=5, zorder=8)
            circ = Circle((com_x,com_y), 0.08, fill=False, edgecolor=acc, lw=1.0, ls='--', zorder=6)
            ax.add_patch(circ)
            ax.text(com_x+0.10, com_y, 'CoM', color=acc, fontsize=8, va='center', fontweight='bold')
            ax.text(0, -0.85, 'CoM offset from geometry centroid', color=acc, fontsize=7, ha='center')

        elif idx == 1:  # Inertia principal axes
            for pts in [body_pts, sp_l_pts, sp_r_pts]:
                ax.fill(pts[:,0], pts[:,1], color='#1a2030', zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            # Three principal axes
            axes_data = [
                (0,0, 0.55,0,'#ff4040','I_xx (max)'),
                (0,0, 0,0.38,'#40ff80','I_yy (mid)'),
                (0,0, 0.32*np.cos(np.radians(35)), 0.32*np.sin(np.radians(35)),'#60d0ff','I_zz (min)'),
            ]
            for ox,oy,dx,dy,c,lbl in axes_data:
                ax.annotate('', xy=(ox+dx, oy+dy), xytext=(ox-dx*0.3, oy-dy*0.3),
                            arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8,
                                            mutation_scale=12), zorder=7)
                ax.annotate('', xy=(ox-dx*0.6, oy-dy*0.6), xytext=(ox+dx*0.2, oy+dy*0.2),
                            arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8,
                                            mutation_scale=12), zorder=7)
                ax.text(ox+dx*1.08, oy+dy*1.08, lbl, color=c, fontsize=7, ha='center')
            # Ellipsoidal inertia body
            ell = matplotlib.patches.Ellipse((0,0), 0.6, 0.45, fill=False,
                                              edgecolor='#6080c0', lw=0.8, ls=':', alpha=0.6, zorder=5)
            ax.add_patch(ell)
            ax.text(0, -0.85, 'Inertia tensor principal axes', color=acc, fontsize=7, ha='center')

        elif idx == 2:  # Surface material / reflectance
            # Draw material zones
            zone_colors = {
                'body': '#2a1a40',
                'solar': '#1a3060',
                'mli': '#603020',
                'antenna': '#404040',
            }
            ax.fill(body_pts[:,0], body_pts[:,1], color=zone_colors['body'], zorder=2)
            for pts in [sp_l_pts, sp_r_pts]:
                ax.fill(pts[:,0], pts[:,1], color=zone_colors['solar'], zorder=2)
            # MLI (multi-layer insulation) strips on body
            for y_strip in np.linspace(-0.15, 0.15, 6):
                ax.fill_between([-0.22, 0.22], [y_strip-0.01], [y_strip+0.01],
                                color='#c09040', alpha=0.5, zorder=3)
            # Reflectance arrows (specular)
            for px,py,ang in [(0.22,0.10,30),(0.22,-0.05,20),(0.22,-0.15,10)]:
                ax.annotate('', xy=(px+0.18*np.cos(np.radians(ang)),
                                     py+0.18*np.sin(np.radians(ang))),
                            xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color='#ffffa0', lw=1.0), zorder=6)
            ax.annotate('', xy=(0.22+0.0, 0.05+0.20), xytext=(0.10, -0.10),
                        arrowprops=dict(arrowstyle='->', color='#ffffa0', lw=1.2, ls='--'), zorder=6)
            # Legend patches
            patches = [
                mpatches.Patch(color='#2a1a40', label='Black body'),
                mpatches.Patch(color='#1a3060', label='Solar cell'),
                mpatches.Patch(color='#c09040', label='MLI blanket'),
            ]
            ax.legend(handles=patches, loc='lower left', fontsize=6, facecolor='#10141f',
                      labelcolor='white', edgecolor='#3050a0')
            ax.text(0, -0.85, 'BRDF material zone classification', color=acc, fontsize=7, ha='center')

        elif idx == 3:  # Thermal IR map
            ax.fill(body_pts[:,0], body_pts[:,1], color='#0a0f1f', zorder=2)
            for pts in [sp_l_pts, sp_r_pts]:
                ax.fill(pts[:,0], pts[:,1], color='#0a0f1f', zorder=2)
            # Thermal gradient
            xx, yy = np.meshgrid(np.linspace(-0.7,0.7,200), np.linspace(-0.25,0.25,80))
            temp = np.zeros_like(xx) - 1
            m_body = (np.abs(xx)<0.22)&(np.abs(yy)<0.18)
            m_spl  = (xx>-0.60)&(xx<-0.24)&(np.abs(yy)<0.06)
            m_spr  = (xx>0.24)&(xx<0.60)&(np.abs(yy)<0.06)
            # Sun-facing side hotter
            temp[m_body] = 200 + 120*(xx[m_body]+0.22)/0.44 + 30*rng.standard_normal(m_body.sum())
            temp[m_spl]  = 80  + 20*rng.standard_normal(m_spl.sum())
            temp[m_spr]  = 220 + 40*rng.standard_normal(m_spr.sum())
            temp_masked = np.ma.masked_where(temp<0, temp)
            im = ax.imshow(temp_masked, extent=[-0.7,0.7,-0.25,0.25], cmap='inferno',
                           vmin=50, vmax=350, alpha=0.9, zorder=3, aspect='auto')
            cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label('T (K)', color='white', fontsize=6)
            cb.ax.yaxis.set_tick_params(color='white', labelsize=5, labelcolor='white')
            # Hot spot marker
            ax.text(0.15, 0.10, '★', color='white', fontsize=10, ha='center', zorder=7)
            ax.text(0, -0.85, 'IR thermal signature mapping', color=acc, fontsize=7, ha='center')

        elif idx == 4:  # Solar radiation pressure
            for pts in [body_pts, sp_l_pts, sp_r_pts]:
                ax.fill(pts[:,0], pts[:,1], color='#1a2030', zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            # Sun direction
            ax.annotate('', xy=(0.68, 0.62), xytext=(0.90, 0.85),
                        arrowprops=dict(arrowstyle='->', color='#ffffa0', lw=2.0), zorder=6)
            ax.text(0.90, 0.90, '☀ Sun', color='#ffffa0', fontsize=8, ha='center')
            # SRP force vectors on panels
            for px, py, sc in [(-0.42, 0, 0.5), (0.42, 0, 0.7)]:
                fx = -0.18*sc; fy = -0.14*sc
                ax.annotate('', xy=(px+fx, py+fy), xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color=acc, lw=1.5), zorder=7)
            # Resultant force
            ax.annotate('', xy=(-0.12, -0.22), xytext=(0.02, -0.02),
                        arrowprops=dict(arrowstyle='->', color='#ff6040', lw=2.0), zorder=7)
            ax.text(-0.14, -0.28, 'F_srp', color='#ff6040', fontsize=8, ha='center')
            ax.text(0, -0.85, 'Solar radiation pressure perturbation', color=acc, fontsize=7, ha='center')

        elif idx == 5:  # Structural stiffness / grapple
            for pts in [body_pts, sp_l_pts, sp_r_pts]:
                ax.fill(pts[:,0], pts[:,1], color='#1a2030', zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            # Stress heatmap on body
            xx2, yy2 = np.meshgrid(np.linspace(-0.22,0.22,60), np.linspace(-0.18,0.18,50))
            stress = np.exp(-3*((xx2-0.10)**2 + (yy2-0.05)**2)) * 0.8 + \
                     np.exp(-4*((xx2+0.05)**2 + (yy2+0.10)**2)) * 0.5
            ax.contourf(xx2, yy2, stress, levels=8, cmap='YlOrRd', alpha=0.55, zorder=3)
            # Grapple fixture markers
            for gx, gy in [(0.18,0.15),(-0.18,0.15),(0.18,-0.15),(-0.18,-0.15)]:
                ax.plot(gx, gy, 'D', color=acc, ms=6, zorder=7)
                circ = Circle((gx,gy), 0.05, fill=False, edgecolor=acc, lw=1.0, zorder=6)
                ax.add_patch(circ)
            ax.text(0, 0.28, 'Grapple Fixtures', color=acc, fontsize=7, ha='center', zorder=8)
            ax.text(0, -0.85, 'Structural load & grapple point map', color=acc, fontsize=7, ha='center')

        ax.set_title(title, color=acc, fontsize=9, fontweight='bold', pad=4)

    fig.suptitle('Physical Cues for Non-Cooperative Target Grasping  |  物理线索',
                 color='white', fontsize=14, fontweight='bold', y=0.96)

    fig.savefig('/workspace/output_2_physical_cues.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved: output_2_physical_cues.png")


# ─────────────────────────────────────────────────────────────
# IMAGE 3 — STATE CUES
# ─────────────────────────────────────────────────────────────

def draw_state_cues():
    fig = plt.figure(figsize=(14, 9), dpi=150)
    fig.patch.set_facecolor('#05060f')

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.04, right=0.97, top=0.88, bottom=0.06)

    panel_titles = [
        'Angular Velocity\n& Tumble Axis',
        'Relative Position\n& Velocity',
        'Attitude Quaternion\nVisualization',
        'Trajectory\nPrediction',
        'Close-Range\nApproach Corridor',
        'Docking Port\nAlignment State',
    ]
    colors_accent = ['#ff6040','#40d0ff','#d060ff','#60ff80','#ffd040','#ff60c0']
    rng = np.random.default_rng(99)

    for idx, (title, acc) in enumerate(zip(panel_titles, colors_accent)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0e1a')
        ax.set_aspect('equal')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a3050')

        xs = rng.uniform(-1, 1, 60); ys = rng.uniform(-1, 1, 60)
        ax.scatter(xs, ys, s=0.5, c='white', alpha=0.3, zorder=0)

        body_pts = np.array([[-0.22,-0.18],[0.22,-0.18],[0.22,0.18],[-0.22,0.18]])
        sp_l_pts = np.array([[-0.60,-0.06],[-0.24,-0.06],[-0.24,0.06],[-0.60,0.06]])
        sp_r_pts = np.array([[0.24,-0.06],[0.60,-0.06],[0.60,0.06],[0.24,0.06]])

        if idx == 0:  # Angular velocity / tumble
            # Draw satellite at slight tilt
            tilt = 15
            t = transforms.Affine2D().rotate_deg(tilt) + ax.transData
            for pts, fc in [(body_pts,'#1a2030'),(sp_l_pts,'#141e2e'),(sp_r_pts,'#141e2e')]:
                poly = plt.Polygon(pts, closed=True, facecolor=fc, edgecolor='#3040a0', lw=0.8,
                                    transform=t, zorder=2)
                ax.add_patch(poly)
            # Tumble arc (rotation about Z)
            theta_arr = np.linspace(0.1, 1.8*np.pi, 200)
            r_arc = 0.55
            ax.plot(r_arc*np.cos(theta_arr), r_arc*np.sin(theta_arr),
                    color=acc, lw=1.8, alpha=0.8, zorder=5)
            ax.annotate('', xy=(r_arc*np.cos(1.8*np.pi), r_arc*np.sin(1.8*np.pi)),
                        xytext=(r_arc*np.cos(1.75*np.pi), r_arc*np.sin(1.75*np.pi)),
                        arrowprops=dict(arrowstyle='->', color=acc, lw=1.8, mutation_scale=12), zorder=6)
            # ω vector
            ax.annotate('', xy=(0.0, 0.68), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color='#ffffa0', lw=2.0,
                                        mutation_scale=14), zorder=7)
            ax.text(0.06, 0.72, 'ω', color='#ffffa0', fontsize=13, zorder=8)
            ax.text(0, -0.85, 'ω ≈ 3.2 °/s  tumble period ≈ 112 s', color=acc, fontsize=7, ha='center')

        elif idx == 1:  # Relative position/velocity
            # Chaser (small)
            chaser = Circle((-0.55, -0.40), 0.08, facecolor='#c8a240', edgecolor='#e8c860', lw=0.8, zorder=4)
            ax.add_patch(chaser)
            ax.text(-0.55, -0.53, 'Chaser', color='#c8a240', fontsize=7, ha='center')
            # Target
            for pts, fc in [(body_pts,'#1a2030'),(sp_l_pts,'#141e2e'),(sp_r_pts,'#141e2e')]:
                ax.fill(pts[:,0], pts[:,1], color=fc, zorder=3)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#3040a0', lw=0.8, zorder=3)
            # Relative position vector r
            ax.annotate('', xy=(0.0, 0.0), xytext=(-0.55, -0.40),
                        arrowprops=dict(arrowstyle='->', color=acc, lw=1.8, mutation_scale=12), zorder=7)
            ax.text(-0.32, -0.28, 'r', color=acc, fontsize=12, fontweight='bold')
            # Velocity vector v_rel
            ax.annotate('', xy=(0.30, 0.20), xytext=(0.0, 0.0),
                        arrowprops=dict(arrowstyle='->', color='#60ff80', lw=1.5, mutation_scale=11), zorder=7)
            ax.text(0.32, 0.24, 'v_rel', color='#60ff80', fontsize=8)
            # Range label
            ax.text(0, -0.85, '|r|=28 m,  |v|=0.12 m/s', color=acc, fontsize=7, ha='center')

        elif idx == 2:  # Attitude quaternion
            # Draw body frame axes
            for pts, fc in [(body_pts,'#1a2030'),(sp_l_pts,'#141e2e'),(sp_r_pts,'#141e2e')]:
                ax.fill(pts[:,0], pts[:,1], color=fc, zorder=2)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            axes_info = [(0.45,0,'#ff4040','x_b'),(0,0.35,'#40ff80','y_b'),
                         (0.28,0.28,'#60d0ff','z_b')]
            for dx,dy,c,lbl in axes_info:
                ax.annotate('', xy=(dx, dy), xytext=(0,0),
                            arrowprops=dict(arrowstyle='->', color=c, lw=1.8, mutation_scale=12), zorder=6)
                ax.text(dx*1.15, dy*1.15, lbl, color=c, fontsize=8, ha='center')
            # Inertial frame (dashed)
            for dx,dy,c,lbl in [(0.42,-0.05,'#ff4040','x_i'),
                                  (0,0.32,'#40ff80','y_i'),(0.25,0.25,'#60d0ff','z_i')]:
                ax.annotate('', xy=(dx-0.12, dy-0.12), xytext=(-0.12,-0.12),
                            arrowprops=dict(arrowstyle='->', color=c, lw=1.0,
                                            mutation_scale=10, linestyle='dashed'), zorder=5)
            # Quaternion text
            q = np.array([0.707, 0.0, 0.0, 0.707])
            ax.text(0, -0.70, f'q = [{q[0]:.3f},  {q[1]:.3f},  {q[2]:.3f},  {q[3]:.3f}]',
                    color='white', fontsize=7, ha='center', family='monospace')
            ax.text(0, -0.85, 'Body-frame quaternion attitude estimate', color=acc, fontsize=7, ha='center')

        elif idx == 3:  # Trajectory prediction
            for pts, fc in [(body_pts,'#1a2030'),(sp_l_pts,'#141e2e'),(sp_r_pts,'#141e2e')]:
                ax.fill(pts[:,0], pts[:,1], color=fc, zorder=3)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            # Past trajectory (solid)
            t_past = np.linspace(-np.pi*0.8, 0, 60)
            tx_p = 0.65*np.cos(t_past)
            ty_p = 0.55*np.sin(t_past)*0.6 - 0.2
            ax.plot(tx_p, ty_p, color=acc, lw=1.5, alpha=0.9, zorder=4)
            # Future prediction (dashed with uncertainty cone)
            t_fut = np.linspace(0, np.pi*0.6, 60)
            tx_f = 0.65*np.cos(t_fut)
            ty_f = 0.55*np.sin(t_fut)*0.6 - 0.2
            ax.plot(tx_f, ty_f, color=acc, lw=1.5, ls='--', alpha=0.7, zorder=4)
            # Uncertainty envelope
            sigma = np.linspace(0, 0.12, 60)
            ax.fill_between(tx_f, ty_f-sigma, ty_f+sigma, color=acc, alpha=0.15, zorder=3)
            # Current position marker
            ax.plot(tx_p[-1], ty_p[-1], 'o', color='white', ms=5, zorder=6)
            ax.text(tx_p[-1]+0.07, ty_p[-1], 'now', color='white', fontsize=7)
            ax.text(0, -0.85, 'Kalman-predicted trajectory ± 3σ', color=acc, fontsize=7, ha='center')

        elif idx == 4:  # Approach corridor
            for pts, fc in [(body_pts,'#1a2030'),(sp_l_pts,'#141e2e'),(sp_r_pts,'#141e2e')]:
                ax.fill(pts[:,0], pts[:,1], color=fc, zorder=3)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            # V-bar approach corridor
            ax.fill([-0.85, -0.22, -0.22, -0.85],
                    [ 0.20,  0.12,  -0.12, -0.20],
                    color=acc, alpha=0.12, zorder=2)
            ax.plot([-0.85,-0.22],[ 0.20, 0.12], color=acc, lw=1.0, ls='--', alpha=0.7)
            ax.plot([-0.85,-0.22],[-0.20,-0.12], color=acc, lw=1.0, ls='--', alpha=0.7)
            # Keep-out zone (red)
            koz = Circle((0,0), 0.32, fill=False, edgecolor='#ff4040',
                          lw=1.2, ls='--', alpha=0.7, zorder=5)
            ax.add_patch(koz)
            ax.text(0, 0.34, 'KOZ', color='#ff4040', fontsize=7, ha='center')
            # Approach direction arrow
            ax.annotate('', xy=(-0.30, 0.0), xytext=(-0.75, 0.0),
                        arrowprops=dict(arrowstyle='->', color=acc, lw=2.0, mutation_scale=14), zorder=6)
            ax.text(-0.54, 0.07, 'V-bar\napproach', color=acc, fontsize=7, ha='center')
            ax.text(0, -0.85, 'Safe approach corridor & keep-out zone', color=acc, fontsize=7, ha='center')

        elif idx == 5:  # Docking port alignment
            for pts, fc in [(body_pts,'#1a2030'),(sp_l_pts,'#141e2e'),(sp_r_pts,'#141e2e')]:
                ax.fill(pts[:,0], pts[:,1], color=fc, zorder=3)
                ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]),
                        color='#2a3a60', lw=0.8, zorder=3)
            # Docking port
            port = Circle((0.22, 0.0), 0.055, facecolor='#304080', edgecolor=acc, lw=1.5, zorder=6)
            ax.add_patch(port)
            ax.text(0.22, 0.10, 'Docking\nPort', color=acc, fontsize=6.5, ha='center')
            # Alignment cross-hair
            for r in [0.10, 0.18, 0.28]:
                c = Circle((0.22, 0.0), r, fill=False, edgecolor=acc, lw=0.6, alpha=0.4, zorder=5)
                ax.add_patch(c)
            ax.plot([0.22-0.30, 0.22+0.30], [0, 0], color=acc, lw=0.8, ls=':', alpha=0.6, zorder=5)
            ax.plot([0.22, 0.22], [-0.30, 0.30], color=acc, lw=0.8, ls=':', alpha=0.6, zorder=5)
            # Gripper approach vector
            ax.annotate('', xy=(0.22-0.07, 0.0), xytext=(-0.30, 0.0),
                        arrowprops=dict(arrowstyle='->', color='#ff6040', lw=2.0, mutation_scale=14), zorder=7)
            # Misalignment angle label
            ax.annotate('', xy=(0.22, 0.055), xytext=(0.22, 0.0),
                        arrowprops=dict(arrowstyle='->', color='#ffffa0', lw=1.0), zorder=7)
            ax.text(0.34, 0.055, 'Δφ=4.2°', color='#ffffa0', fontsize=7)
            ax.text(0, -0.85, 'Port alignment error & lateral offset', color=acc, fontsize=7, ha='center')

        ax.set_title(title, color=acc, fontsize=9, fontweight='bold', pad=4)

    fig.suptitle('State Cues for Non-Cooperative Target Grasping  |  状态线索',
                 color='white', fontsize=14, fontweight='bold', y=0.96)

    fig.savefig('/workspace/output_3_state_cues.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved: output_3_state_cues.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    draw_robotic_arm_base()
    draw_geometric_cues()
    draw_physical_cues()
    draw_state_cues()
    print("All four images generated successfully.")
