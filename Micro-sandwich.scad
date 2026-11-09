
// Micro‑Sandwich™ — Parametric OpenSCAD Template (v0.2)
// Units: millimeters unless noted. Use um(x) for microns.
// Intended for concept visualization and quick iteration. Not a final manufacturing drawing.
//
// Layers (bottom→top):
//  - Bottom coverslip (#1.5H, ~0.17 mm)
//  - Cell gap (set by spacer beads/frame) — default 20 µm
//  - Top coverslip (#1.5H)
//  - Actuated flexible dome (TPU/PET), optional
//
// Key parameters to tune: gap_um, dome_diameter, dome_lift, window_len, window_wid
//
// HOW TO USE:
//  - Load in OpenSCAD → F5 preview, F6 render
//  - Adjust parameters below and export STL/DXF for quick jigs or gasket cutting
//  - For real-world builds, replace the "spacer_frame" with PSA gasket or spacer beads
//
// --------------------------------------------------------------

/* ---------- Parameters ---------- */
// Microscale helper
function um(x) = x / 1000.0;   // microns → mm

// Imaging window size (visible field)
window_len = 18;      // mm
window_wid = 8;       // mm

// Glass thickness (#1.5H typical ≈ 0.17 mm)
th_glass = 0.17;      // mm (both bottom and top)

// Target gap between glasses (set by beads/gasket)
gap_um    = 20;       // microns
gap_mm    = um(gap_um);

// Rigid rim around the window (top “lid” frame)
rim_w     = 2.0;      // mm rim width
rim_th    = 0.30;     // mm rigid rim thickness

// Flexible actuation dome (optional)
dome_diameter = 12;   // mm
dome_lift     = 0.15; // mm (≈150 µm max central deflection)
mem_th        = 0.10; // mm membrane thickness

// Safety margins for outside size
outer_margin = 4;     // mm around the window

/* ---------- Derived ---------- */
plate_len = window_len + 2*outer_margin;
plate_wid = window_wid + 2*outer_margin;
total_th  = th_glass + gap_mm + th_glass + mem_th + rim_th;

/* ---------- Modules ---------- */
module coverslip(len, wid, th){
    translate([0,0,0]) cube([len, wid, th], center=false);
}

module spacer_frame(len, wid, thickness, frame_w){
    // Simple rectangular frame that defines the gap (for visualization)
    difference(){
        cube([len, wid, thickness], center=false);
        translate([frame_w, frame_w, -0.01])
            cube([len-2*frame_w, wid-2*frame_w, thickness+0.02], center=false);
    }
}

module top_rim_with_window(len, wid, rim_w, rim_th){
    // Rigid rim with a window in the center (no membrane here)
    difference(){
        cube([len, wid, rim_th], center=false);
        translate([rim_w, rim_w, -0.01])
            cube([len-2*rim_w, wid-2*rim_w, rim_th+0.02], center=false);
    }
}

module dome(d, t, lift){
    // Axisymmetric dome approximated by a shallow spherical cap
    // Rendered as a solid lens-like cap of thickness 't' and height 'lift'
    r = d/2;
    // Thickness is applied at base to avoid zero-thickness issues
    union(){
        // base membrane disk
        cylinder(h=t, r=r, $fn=180);
        // raised cap
        translate([0,0,t])
            scale([1,1,lift/(r/2+0.0001)])
                sphere(r=r/2, $fn=180);
    }
}

/* ---------- Assembly ---------- */
module micro_sandwich(){
    // Coordinate origin at bottom-left of bottom coverslip

    // Bottom coverslip
    color([0.9,0.9,1.0,0.4]) // translucent hint
    translate([0,0,0])
        coverslip(plate_len, plate_wid, th_glass);

    // Spacer definition (beads OR frame). Here we show a frame at the perimeter.
    // Replace with discrete spheres to visualize beads if preferred.
    color([0.8,0.8,0.8,0.8])
    translate([0,0,th_glass])
        spacer_frame(plate_len, plate_wid, gap_mm, outer_margin - 0.5);

    // Top coverslip sitting on the spacer
    color([0.9,0.9,1.0,0.4])
    translate([0,0,th_glass + gap_mm])
        coverslip(plate_len, plate_wid, th_glass);

    // Top rigid rim (with central window) glued above the top coverslip
    color([0.85,0.85,0.85,1.0])
    translate([0,0,th_glass + gap_mm + th_glass])
        top_rim_with_window(plate_len, plate_wid, rim_w, rim_th);

    // Flexible actuation dome, centered above the window
    // Move Z a tiny epsilon above rim to avoid z-fighting in preview
    color([0.75,0.75,0.75,0.7])
    translate([plate_len/2, plate_wid/2, th_glass + gap_mm + th_glass + rim_th + 0.01])
        dome(dome_diameter, mem_th, dome_lift);
}

// Optional: bead visualization at corners (comment in if desired)
// module bead(d_um){ sphere(r=um(d_um)/2, $fn=60); }
// translate([outer_margin, outer_margin, th_glass + um(10)]) bead(20);
// translate([plate_len-outer_margin, outer_margin, th_glass + um(10)]) bead(20);
// translate([outer_margin, plate_wid-outer_margin, th_glass + um(10)]) bead(20);
// translate([plate_len-outer_margin, plate_wid-outer_margin, th_glass + um(10)]) bead(20);

/* ---------- Call the assembly ---------- */
micro_sandwich();
