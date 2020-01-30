use super::binary::*;
use super::unary::*;
use super::*;
use quickcheck::quickcheck;

#[test]
fn sat() {
    let mut sat = Solver::new();
    let a = sat.new_lit();
    let b = sat.new_lit();
    // sat.add_clause(&[a,b]);
    sat.add_clause(once(a).chain(once(b)));
    match sat.solve() {
        Ok(m) => {
            println!("a: {:?}", m.value(&a));
            println!("b: {:?}", m.value(&b));
        }
        Err(_) => panic!(),
    };
}

#[test]
fn unsat() {
    let mut sat = Solver::new();
    let a = sat.new_lit();
    let b = sat.new_lit();
    // sat.add_clause(&[a]);
    // sat.add_clause(&[b]);
    // sat.add_clause(&[!b]);
    sat.add_clause(once(a));
    sat.add_clause(once(b));
    sat.add_clause(once(!b));
    let sol = sat.solve();
    assert_eq!(sol.is_err(), true);
}

#[test]
fn unsat2() {
    use std::iter::empty;
    let mut sat = Solver::new();
    sat.add_clause(empty());
    assert_eq!(sat.solve().is_err(), true);
}

#[test]
fn sat2() {
    let mut sat = Solver::new();
    let a = sat.new_lit();
    assert_eq!(sat.solve().is_err(), false);
    assert_eq!(sat.solve_under_assumptions(vec![a]).is_err(), false);
    sat.add_clause(once(a));
    assert_eq!(sat.solve().is_err(), false);
    assert_eq!(sat.solve_under_assumptions(vec![!a]).is_err(), true);
    sat.add_clause(vec![!a]);
    assert_eq!(sat.solve().is_err(), true);
}

#[test]
fn xor() {
    let mut sat = Solver::new();
    let a = sat.new_lit();
    let b = sat.new_lit();
    let c = sat.new_lit();
    let d = sat.new_lit();
    let x = sat.xor_literal(vec![a, !b, c, d]);
    sat.add_clause(vec![x]);
    loop {
        let (av, bv, cv, dv);
        match sat.solve() {
            Ok(model) => {
                av = model.value(&a);
                bv = model.value(&b);
                cv = model.value(&c);
                dv = model.value(&d);
                println!("MODEL: a={}\tb={}\tc={}\td={}", av, bv, cv, dv);
                assert_eq!(true, av ^ (!bv) ^ cv ^ dv);
            }
            _ => {
                break;
            }
        };

        sat.add_clause(
            vec![av, bv, cv, dv]
                .into_iter()
                .zip(vec![a, b, c, d])
                .map(|(v, x)| if v { !x } else { x }),
        );
    }
}

#[test]
fn unary_1() {
    let mut sat = Solver::new();
    let a = Unary::new(&mut sat, 100);
    let b = Unary::new(&mut sat, 200);
    sat.greater_than(&a, &Unary::constant(50));
    sat.less_than(&a.mul_const(2), &b);

    match sat.solve() {
        Ok(model) => {
            let av = model.value(&a);
            println!("A value: {}", av);
            let bv = model.value(&b);
            println!("B value: {}", bv);
            assert!(av > 50);
            assert!(bv > av);
        }
        _ => panic!(),
    }
}

#[test]
fn graph_color() {
    use symbolic::*;
    let mut coloring = Solver::new();

    #[derive(PartialEq, Eq, Debug, PartialOrd, Ord)]
    enum Color {
        Red,
        Green,
        Blue,
    };

    let n_nodes = 5;
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (3, 1), (4, 0), (4, 2)];
    let colors = (0..n_nodes)
        .map(|_| Symbolic::new(&mut coloring, vec![Color::Red, Color::Green, Color::Blue]))
        .collect::<Vec<_>>();
    for (n1, n2) in edges {
        coloring.not_equal(&colors[n1], &colors[n2]);
    }
    match coloring.solve() {
        Ok(model) => {
            for i in 0..n_nodes {
                println!("Node {}: {:?}", i, model.value(&colors[i]));
            }
        }
        Err(()) => {
            println!("No solution.");
        }
    }
}

#[test]
fn take_more_than_len() {
    let mut iter = vec![1, 2, 3].into_iter().take(9999);
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), None);
}

#[test]
fn factorization_unary() {
    let mut sat = Solver::new();
    let a = Unary::new(&mut sat, 20);
    let b = Unary::new(&mut sat, 20);
    let c = a.mul(&mut sat, &b);
    sat.equal(&c, &Unary::constant(209));

    println!("Solving {:?}", sat);
    match sat.solve() {
        Ok(model) => {
            println!("{}*{}=209", model.value(&a), model.value(&b));
            assert_eq!(model.value(&a) * model.value(&b), 209);
        }
        Err(()) => {
            println!("No solution.");
        }
    }
}

#[test]
fn factorization_binary() {
    let mut sat = Solver::new();
    let a = Binary::new(&mut sat, 1000);
    let b = Binary::new(&mut sat, 1000);
    let c = a.mul(&mut sat, &b);
    sat.equal(&c, &Binary::constant(36863));

    println!("Solving {:?}", sat);
    match sat.solve() {
        Ok(model) => {
            println!("{}*{}=36863", model.value(&a), model.value(&b));
        }
        Err(()) => {
            println!("No solution.");
        }
    }
}

#[test]
fn factorization_binary_large() {
    let mut sat = Solver::new();
    let a = Binary::new(&mut sat, 10000);
    let b = Binary::new(&mut sat, 10000);
    let c = a.mul(&mut sat, &b);
    sat.equal(&c, &Binary::constant(3686301));

    println!("Solving {:?}", sat);
    match sat.solve() {
        Ok(model) => {
            println!("{}*{}=3686301", model.value(&a), model.value(&b));
        }
        Err(()) => {
            println!("No solution.");
        }
    }
}

#[test]
fn binary_ord() {
    let mut sat = Solver::new();
    let a = Binary::new(&mut sat, 2_usize.pow(16));
    let b = Binary::new(&mut sat, 123123123123);
    let c = Binary::new(&mut sat, 1231231231239);

    sat.less_than(&Binary::constant(30), &a);
    sat.less_than(&a, &Binary::constant(90));
    sat.less_than(&Binary::constant(15), &b);
    sat.less_than(&b, &Binary::constant(17));
    let d = a.add(&mut sat, &b);
    let e = a.add(&mut sat, &Binary::constant(2));
    // sat.less_than(&a, &b);
    //        sat.less_than(&Binary::constant(100001), &b);
    //        sat.greater_than(&Binary::constant(100003), &b);
    //        let d = a.add(&mut sat, &Binary::constant(100));
    //        sat.equal(&d, &b);
    // sat.greater_than(&c, &b);
    // sat.less_than_equal(&c, &Binary::constant(100100));

    println!("Solving {:?}", sat);
    match sat.solve() {
        Ok(m) => {
            println!(
                "a={}, b={}, c={}, d={}, e={}",
                m.value(&a),
                m.value(&b),
                m.value(&c),
                m.value(&d),
                m.value(&e)
            );
            // assert_eq!(m.value(&b), 100002);
        }
        Err(()) => panic!(),
    }
}

quickcheck! {
    fn binary_comparison(y :usize) -> bool {
        let mut s = Solver::new();
        let c = Binary::constant(y);
        let x = Binary::new(&mut s, 2*y);
        let lte = s.new_lit();
        let lt = s.new_lit();
        let gt = s.new_lit();
        let gte = s.new_lit();

        s.less_than_equal_or(vec![!lte], &c, &x);
        s.greater_than_equal_or(vec![!gte], &c, &x);
        s.less_than_or(vec![!lt], &c, &x);
        s.greater_than_or(vec![!gt], &c, &x);

        let m1 = s.solve_under_assumptions(vec![lte]).unwrap();
        if !(y <= m1.value(&x)) { return false; }

        let m2 = s.solve_under_assumptions(vec![gte]).unwrap();
        if !(y >= m2.value(&x)) { return false; }

        let m5 = s.solve_under_assumptions(vec![gte, lte]).unwrap();
        if !(y == m5.value(&x)) { return false; }

        if y > 0 {
            let m3 = s.solve_under_assumptions(vec![lt]).unwrap();
            if !(y < m3.value(&x)) { return false; }

            let m4 = s.solve_under_assumptions(vec![gt]).unwrap();
            if !(y > m4.value(&x)) { return false; }

            if !s.solve_under_assumptions(vec![gt, lt]).is_err() { return false; };
        }

        true
    }
}

quickcheck! {
    fn const_binary_eq(xs :Vec<usize>) -> bool {
        let mut sat = Solver::new();
        let xs = xs.into_iter().map(|x| {
            //println!("CONST BINARY EQ {}", x);
            let b = Binary::new(&mut sat, x);
            sat.equal(&b, &Binary::constant(x));
            (x,b)
        }).collect::<Vec<_>>();

        match sat.solve() {
            Ok(m) => {
                for (x,b) in xs {
                    assert_eq!(x, m.value(&b));
                }
            },
            _ => panic!(),
        };
        true
    }
}

quickcheck! {
    fn xor_odd_constant(lits :Vec<bool>) -> bool {
        // The xor literal function returns the odd parity bit
        // which is a constant when the input is a list of constants
        let mut sat = Solver::new();
        let f = sat.xor_literal(lits.iter()
                        .map(|_| false.into())) == false.into();
        let t = sat.xor_literal(lits.iter()
                        .map(|_| true.into())) == (lits.len() % 2 == 1).into();
        t && f
    }
}

quickcheck! {
    fn xor_literal_lits(lits :Vec<bool>) -> bool {
        let mut sat = Solver::new();
        if lits.is_empty() { return true; }
        let lits = lits.iter().map(|_| sat.new_lit()).collect::<Vec<_>>();
        let xor = sat.xor_literal(lits.iter().cloned());
        sat.add_clause(vec![xor]); // assert odd parity of list of literals

        match sat.solve() {
            Ok(m) => {
                let model_parity = lits.iter().map(|x| {
                    if m.value(x) { 1usize } else {0usize }
                }).sum::<usize>() % 2;
                assert_eq!(model_parity, 1);
            },
            Err(()) => panic!(),
        };
        true
    }
}

quickcheck! {
    fn xor_literal(lits :Vec<bool>, consts :Vec<bool>) -> bool {
        let mut sat = Solver::new();
        let lits = lits.iter().map(|_| sat.new_lit()).collect::<Vec<_>>();
        let expr = consts.iter()
            .map(|x| (*x).into()).chain(lits.into_iter()).collect::<Vec<_>>();
        let xor = sat.xor_literal(expr.iter().cloned());

        match sat.solve() {
            Ok(m) => {
                let model_parity = expr.iter().map(|x| {
                    //println!(" {:?} -> {:?}", x, m.value(x));
                    if m.value(x) { 1usize } else { 0usize }
                })
                    .sum::<usize>() % 2 == 1;
                assert_eq!(model_parity, m.value(&xor));
            }
            Err(()) => panic!(),
        };
        true
    }
}

quickcheck! {
    fn parity(xs :Vec<bool>) -> bool {
        let mut sat = Solver::new();
        let parity = xs.iter()
            .map(|x| if *x { 1usize } else { 0usize })
            .sum::<usize>() % 2 == 1;
        let lits = xs.iter().map(|x| {
            let lit = sat.new_lit();
            sat.equal(&lit, &(*x).into());
            lit
        }).collect::<Vec<_>>();
        sat.assert_parity(lits, parity);

        match sat.solve() {
            Err(()) => panic!(),
            _ => {},
        };
        true
    }
}

quickcheck! {
    fn const_bool_equal(xs :Vec<bool>) -> bool {
        let mut sat = Solver::new();
        let xs = xs.into_iter().map(|x| {
            let y = sat.new_lit();
            sat.equal(&y,&x.into());
            (x,y)
        }).collect::<Vec<_>>();

        match sat.solve() {
            Ok(m) => {
                for (x,y) in xs {
                    assert_eq!(x, m.value(&y));
                }
            },
            _ => panic!(),
        };
        true
    }
}

quickcheck! {
    fn const_bool_addclause(xs :Vec<bool>) -> bool {
        let mut sat = Solver::new();
        let xs = xs.into_iter().map(|x| {
            let y = sat.new_lit();
            sat.add_clause(vec![y, (!x).into()]); // x -> y == y, !x
            sat.add_clause(vec![x.into(), !y]); // y -> x == x, !y
            (x,y)
        }).collect::<Vec<_>>();

        match sat.solve() {
            Ok(m) => {
                for (x,y) in xs {
                    assert_eq!(x, m.value(&y));
                }
            },
            _ => panic!(),
        };
        true
    }
}
