
let fun [N] (A : [N][64]f32) : [N][64]f32 =
    map (\row ->
        map (\elm ->
            elm * elm
        ) row 
        |> scan (+) 0
    ) A

