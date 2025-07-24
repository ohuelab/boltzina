MGL_PATH=/gs/bs/tga-furui/apps/mgltools_x86_64Linux2_1.5.7
CONFIG_FILE=test_data/KIF11/docking/input.txt
LIGAND_CHAIN=B
INPUT_RECEPTOR=/home/6/uc02086/workspace-kf/fold/boltzina/test_data/KIF11/boltz_out/boltz_results_base_config/predictions/base_config/base_config_model_0_complex.pdb
DOCKING_DIR=/home/6/uc02086/workspace-kf/fold/boltzina/test_data/KIF11/docking
uv_activate boltz2_env

boltz predict --out_dir test_data/KIF11/boltz_out test_data/KIF11/base_config.yaml

# cif->pdb->_complex.pdb

echo $MGL_PATH/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py
LD_LIBRARY_PATH=${MGL_PATH}/lib $MGL_PATH/bin/pythonsh \
    $MGL_PATH/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py \
    -r $INPUT_RECEPTOR \
    -o $DOCKING_DIR
cd /gs/bs/tga-furui/workspace/fold/boltzina/test_data/KIF11/active_mols && LD_LIBRARY_PATH=${MGL_PATH}/lib $MGL_PATH/bin/pythonsh \
    $MGL_PATH/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py \
    -l CHEMBL1163892.sdf \
    -o /home/6/uc02086/workspace-kf/fold/boltzina/test_data/KIF11/docking/ligand.pdbqt
python /home/6/uc02086/workspace-kf/fold/boltzina/src/docking/get_centerofmass.py \
    /home/6/uc02086/workspace-kf/fold/boltzina/test_data/KIF11/boltz_out/boltz_results_base_config/predictions/base_config/base_config_model_0.pdb \
    --chain B
vina --config test_data/KIF11/docking/input.txt \
     --receptor test_data/KIF11/docking/receptor.pdbqt --ligand test_data/KIF11/docking/ligand.pdbqt --out test_data/KIF11/docking/docking_out.pdbqt

obabel test_data/KIF11/docking/docking_out.pdbqt -m -O test_data/KIF11/docking/docking_outs/docking_out_.pdb
pdb_chain -B test_data/KIF11/docking/docking_outs/docking_out_1.pdb  | pdb_rplresname -"<0>":MOL | pdb_tidy > test_data/KIF11/docking/docking_outs/docking_out_1_B.pdb
pdb_merge test_data/KIF11/boltz_out/boltz_results_base_config/predictions/base_config/base_config_model_0_complex.pdb test_data/KIF11/docking/docking_outs/docking_out_1_B.pdb | pdb_tidy > test_data/KIF11/docking/merge_1.pdb

maxit -input test_data/KIF11/docking/merge_1.pdb -output test_data/KIF11/docking/merge_1.cif -o 1
maxit -input test_data/KIF11/docking/merge_1.cif -output test_data/KIF11/docking/merge_1_fix.cif -o 8
