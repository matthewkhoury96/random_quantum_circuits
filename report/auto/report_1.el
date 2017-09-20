(TeX-add-style-hook
 "report_1"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "margin=1in") ("natbib" "numbers") ("hyperref" "colorlinks" "allcolors=blue")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "geometry"
    "amsmath"
    "amsthm"
    "amsfonts"
    "amssymb"
    "mathtools"
    "color"
    "graphicx"
    "overpic"
    "mathrsfs"
    "enumitem"
    "braket"
    "parskip"
    "natbib"
    "hyperref"
    "caption")
   (LaTeX-add-labels
    "collision_probability"
    "cp_conjecture"
    "conjecture_1"
    "conjecture_2"
    "pauli_1"
    "pauli_3"
    "pauli_2"
    "shorthand_1"
    "isomorphism"
    "stab_1"
    "cond_1"
    "cond_2"
    "stab_shorthand"
    "clifford_gates"
    "cg_1"
    "cg_2"
    "p_cond_1"
    "p_cond_2"
    "proj_1"
    "proj_2"
    "proj_3"
    "prelim_1"
    "prelim_2"
    "prelim_4"
    "prelim_5"
    "simplifying_cp"
    "simplify_1"
    "simplify_2"
    "simplify_3"
    "simplify_4"
    "simplify_5"
    "simplify_6"
    "simplify_7"
    "simplify_8"
    "simplify_9"
    "algo_1"
    "algo_step"
    "algo_cp"
    "ker_1"
    "algo_proof_1"
    "algo_proof_2"
    "proto_1"
    "p_first"
    "p_middle"
    "p_last"
    "proto_2"
    "1d_lattice"
    "fig_1d"
    "k_1d"
    "d_star_1d"
    "fig_2d"
    "k_2d"
    "d_star_2d"
    "fig_3d"
    "k_3d"
    "d_star_3d"
    "fig_cg"
    "k_cg"
    "N_star_cg"
    "fig_1d_std"
    "fig_2d_std"
    "fig_3d_std"
    "fig_cg_std"
    "fig_steady_state")
   (LaTeX-add-bibliographies
    "mybib")
   (LaTeX-add-amsthm-newtheorems
    "thm"
    "claim"
    "algorithm"
    "cor"
    "lem"
    "prop"
    "proto"
    "con"
    "remark"
    "observation"
    "example"
    "conjecture"
    "dfn"))
 :latex)

