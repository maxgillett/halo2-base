use std::marker::PhantomData;

use crate::{
    gates::GateInstructions,
    AssignedValue, Context,
    QuantumCell::{Constant, Existing},
};
use halo2_proofs::{halo2curves::FieldExt, plonk::Error};
// taken from https://github.com/scroll-tech/halo2-snark-aggregator/tree/main/halo2-snark-aggregator-api/src/hash
use poseidon::{SparseMDSMatrix, Spec, State};

struct PoseidonState<F: FieldExt, A: GateInstructions<F>, const T: usize, const RATE: usize> {
    s: [AssignedValue<F>; T],
    _marker: PhantomData<A>,
}

impl<F: FieldExt, A: GateInstructions<F>, const T: usize, const RATE: usize>
    PoseidonState<F, A, T, RATE>
{
    fn x_power5_with_constant(
        ctx: &mut Context<'_, F>,
        chip: &A,
        x: &AssignedValue<F>,
        constant: &F,
    ) -> Result<AssignedValue<F>, Error> {
        let x2 = chip.mul(ctx, &Existing(x), &Existing(x))?;
        let x4 = chip.mul(ctx, &Existing(&x2), &Existing(&x2))?;
        chip.mul_add(ctx, &Existing(x), &Existing(&x4), &Constant(*constant))
    }

    fn sbox_full(
        &mut self,
        ctx: &mut Context<'_, F>,
        chip: &A,
        constants: &[F; T],
    ) -> Result<(), Error> {
        for (x, constant) in self.s.iter_mut().zip(constants.iter()) {
            *x = Self::x_power5_with_constant(ctx, chip, x, constant)?;
        }
        Ok(())
    }

    fn sbox_part(&mut self, ctx: &mut Context<'_, F>, chip: &A, constant: &F) -> Result<(), Error> {
        let x = &mut self.s[0];
        *x = Self::x_power5_with_constant(ctx, chip, x, constant)?;

        Ok(())
    }

    fn absorb_with_pre_constants(
        &mut self,
        ctx: &mut Context<'_, F>,
        chip: &A,
        inputs: Vec<AssignedValue<F>>,
        pre_constants: &[F; T],
    ) -> Result<(), Error> {
        assert!(inputs.len() < T);
        let offset = inputs.len() + 1;

        (_, _, self.s[0]) = chip.inner_product(
            ctx,
            &[Constant(pre_constants[0])]
                .into_iter()
                .chain(inputs.iter().map(|a| Existing(a)))
                .collect(),
            &vec![Constant(F::one()); inputs.len() + 1],
        )?;

        for ((x, constant), input) in self
            .s
            .iter_mut()
            .skip(1)
            .zip(pre_constants.iter().skip(1))
            .zip(inputs.iter())
        {
            *x = if *constant == F::zero() {
                chip.add(ctx, &Existing(x), &Existing(input))?
            } else {
                chip.inner_product(
                    ctx,
                    &vec![Existing(x), Existing(input), Constant(*constant)],
                    &vec![Constant(F::one()), Constant(F::one()), Constant(F::one())],
                )?
                .2
            };
        }

        for (i, (x, constant)) in self
            .s
            .iter_mut()
            .skip(offset)
            .zip(pre_constants.iter().skip(offset))
            .enumerate()
        {
            *x = chip.add(
                ctx,
                &Existing(x),
                &Constant(if i == 0 {
                    *constant + F::one()
                } else {
                    *constant
                }),
            )?;
        }

        Ok(())
    }

    fn apply_mds(
        &mut self,
        ctx: &mut Context<'_, F>,
        chip: &A,
        mds: &[[F; T]; T],
    ) -> Result<(), Error> {
        let res = mds
            .iter()
            .map(|row| {
                let (_, _, sum) = chip.inner_product(
                    ctx,
                    &self.s.iter().map(|a| Existing(a)).collect(),
                    &row.iter().map(|c| Constant(*c)).collect(),
                )?;
                Ok(sum)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        self.s = res.try_into().unwrap();

        Ok(())
    }

    fn apply_sparse_mds(
        &mut self,
        ctx: &mut Context<'_, F>,
        chip: &A,
        mds: &SparseMDSMatrix<F, T, RATE>,
    ) -> Result<(), Error> {
        let (_, _, sum) = chip.inner_product(
            ctx,
            &self.s.iter().map(|a| Existing(a)).collect(),
            &mds.row().iter().map(|c| Constant(*c)).collect(),
        )?;
        let mut res = vec![sum];

        for (e, x) in mds.col_hat().iter().zip(self.s.iter().skip(1)) {
            res.push(chip.mul_add(ctx, &Existing(&self.s[0]), &Constant(*e), &Existing(x))?);
        }

        for (x, new_x) in self.s.iter_mut().zip(res.into_iter()) {
            *x = new_x
        }

        Ok(())
    }
}

pub struct PoseidonChip<F: FieldExt, A: GateInstructions<F>, const T: usize, const RATE: usize> {
    init_state: [AssignedValue<F>; T],
    state: PoseidonState<F, A, T, RATE>,
    spec: Spec<F, T, RATE>,
    absorbing: Vec<AssignedValue<F>>,
}

impl<F: FieldExt, A: GateInstructions<F>, const T: usize, const RATE: usize>
    PoseidonChip<F, A, T, RATE>
{
    pub fn new(ctx: &mut Context<'_, F>, chip: &A, r_f: usize, r_p: usize) -> Result<Self, Error> {
        let init_state = State::<F, T>::default()
            .words()
            .into_iter()
            .map(|x| {
                Ok(chip
                    .assign_region(ctx, vec![Constant(x)], vec![], None)?
                    .pop()
                    .unwrap())
            })
            .collect::<Result<Vec<AssignedValue<F>>, Error>>()?;
        Ok(Self {
            spec: Spec::new(r_f, r_p),
            init_state: init_state.clone().try_into().unwrap(),
            state: PoseidonState {
                s: init_state.try_into().unwrap(),
                _marker: PhantomData,
            },
            absorbing: Vec::new(),
        })
    }

    pub fn clear(&mut self) {
        self.state = PoseidonState {
            s: self.init_state.clone(),
            _marker: PhantomData,
        };
        self.absorbing.clear();
    }

    pub fn update(&mut self, elements: &[AssignedValue<F>]) {
        self.absorbing.extend_from_slice(elements);
    }

    pub fn squeeze(
        &mut self,
        ctx: &mut Context<'_, F>,
        chip: &A,
    ) -> Result<AssignedValue<F>, Error> {
        let mut input_elements = vec![];
        input_elements.append(&mut self.absorbing);

        let mut padding_offset = 0;

        for chunk in input_elements.chunks(RATE) {
            padding_offset = RATE - chunk.len();
            self.permutation(ctx, chip, chunk.to_vec())?;
        }

        if padding_offset == 0 {
            self.permutation(ctx, chip, vec![])?;
        }

        Ok(self.state.s[1].clone())
    }

    fn permutation(
        &mut self,
        ctx: &mut Context<'_, F>,
        chip: &A,
        inputs: Vec<AssignedValue<F>>,
    ) -> Result<(), Error> {
        let r_f = self.spec.r_f() / 2;
        let mds = &self.spec.mds_matrices().mds().rows();

        let constants = &self.spec.constants().start();
        self.state
            .absorb_with_pre_constants(ctx, chip, inputs, &constants[0])?;
        for constants in constants.iter().skip(1).take(r_f - 1) {
            self.state.sbox_full(ctx, chip, constants)?;
            self.state.apply_mds(ctx, chip, mds)?;
        }

        let pre_sparse_mds = &self.spec.mds_matrices().pre_sparse_mds().rows();
        self.state.sbox_full(ctx, chip, constants.last().unwrap())?;
        self.state.apply_mds(ctx, chip, pre_sparse_mds)?;

        let sparse_matrices = &self.spec.mds_matrices().sparse_matrices();
        let constants = &self.spec.constants().partial();
        for (constant, sparse_mds) in constants.iter().zip(sparse_matrices.iter()) {
            self.state.sbox_part(ctx, chip, constant)?;
            self.state.apply_sparse_mds(ctx, chip, sparse_mds)?;
        }

        let constants = &self.spec.constants().end();
        for constants in constants.iter() {
            self.state.sbox_full(ctx, chip, constants)?;
            self.state.apply_mds(ctx, chip, mds)?;
        }
        self.state.sbox_full(ctx, chip, &[F::zero(); T])?;
        self.state.apply_mds(ctx, chip, mds)?;

        Ok(())
    }
}
