#!/usr/local/bin/bash







./plot_expanding_window.py --region 'GREATER SYDNEY'   --step_months=3 --agg median --order 2,0,1 -f expand_w_greater_sydney.pdf
./plot_expanding_window.py --region 'GREATER BRISBANE'   --step_months=3 --agg median --order 2,0,1 -f expand_w_greater_brisbane.pdf
./plot_expanding_window.py --region 'GREATER ADELAIDE'   --step_months=3 --agg median --order 2,0,1 -f expand_w_greater_adelaide.pdf
./plot_expanding_window.py --region 'REST OF QLD'   --step_months=3 --agg median --order 1,0,2 -f expand_w_rest_of_qld.pdf

./plot_expanding_window.py --region 'GREATER MELBOURNE'   --step_months=3 --agg median --order 2,0,2 -f expand_w_greater_melbourne.pdf

./plot_expanding_window.py --region 'REST OF NSW'   --step_months=3 --agg median --order 2,0,0 -f expand_w_rest_of_nsw.pdf
./plot_expanding_window.py --region 'GREATER PERTH'   --step_months=3 --agg median --order 2,0,1 -f expand_w_perth.pdf
./plot_expanding_window.py --region 'GREATER HOBART'   --step_months=3 --agg median --order 1,0,1 -f expand_w_hobart.pdf
./plot_expanding_window.py --region 'REST OF VIC.'   --step_months=3 --agg median --order 2,0,0 -f expand_w_rest_of_vic.pdf
./plot_expanding_window.py --region 'REST OF WA'   --step_months=3 --agg median --order 2,0,0 -f expand_w_rest_of_wa.pdf
./plot_expanding_window.py --region 'REST OF SA'   --step_months=3 --agg median --order 2,0,0 -f expand_w_rest_of_sa.pdf

